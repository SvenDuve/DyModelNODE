
# using Flux, DifferentialEquations#, DiffEqFlux
using Parameters
using Conda
using PyCall
using Flux
using Flux.Optimise
#using DiffEqFlux
using DifferentialEquations
using SciMLSensitivity
using ForwardDiff
#using DiffEqSensitivity



export greet,
    HyperParameter,
    agent



@with_kw mutable struct EnvParameter
    # Dimensions
    ## Actions
    action_size::Int =                      1
    action_bound::Float32 =                 1.f0
    action_bound_high::Array{Float32} =     [1.f0]
    action_bound_low::Array{Float32} =      [-1.f0]
    ## States
    state_size::Int =                       1
    state_bound_high::Array{Float32} =      [1.f0]
    state_bound_low::Array{Float32} =       [1.f0]
end


@with_kw mutable struct HyperParameter
    # Buffer size
    buffer_size::Int =                      1000000
    # Exploration
    expl_noise::Float32 =                   0.2f0
    noise_clip::Float32 =                   1.f0
    # Training Metrics
    training_episodes::Int =                5
    maximum_episode_length::Int =           10
    train_start:: Int =                     1
    batch_size::Int =                       64
    # Metrics
    episode_reward::Array{Float32} =        []
    critic_loss::Array{Float32} =           [0.f0]
    actor_loss::Array{Float32} =            [0.f0]
    episode_steps::Array{Int} =             []
    # Discount
    γ::Float32 =                            0.99f0
    # Learning Rates
    critic_η::Float64 =                     0.001
    actor_η::Float64 =                      0.001
    # Agents
    store_frequency::Int =                  100
    trained_agents =                        []
    # Mean Predition Error
    mpe =                                   []
end




# Define the experience replay buffer
mutable struct ReplayBuffer
    capacity::Int
    memory::Vector{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}
    pos::Int
end

# outer constructor for the Replay Buffer
function ReplayBuffer(capacity::Int)
    memory = []
    return ReplayBuffer(capacity, memory, 1)
end


function remember(buffer::ReplayBuffer, state, action, reward, next_state, done)
    if length(buffer.memory) < buffer.capacity
        push!(buffer.memory, (state, action, reward, next_state, done))
    else
        buffer.memory[buffer.pos] = (state, action, reward, next_state, done)
    end
    buffer.pos = mod1(buffer.pos + 1, buffer.capacity)
end


function sample(buffer::ReplayBuffer, batch_size::Int)
    batch = rand(buffer.memory, batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end

    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end



# Define the action, actor_loss, and critic_loss functions
function action(model, state, train, ep, hp)
    if train
        a = model(state) .+ clamp.(rand(Normal{Float32}(0.f0, hp.expl_noise), size(ep.action_size)), -hp.noise_clip, hp.noise_clip)
        return clamp.(a, ep.action_bound_low, ep.action_bound_high)
    else
        return model(state)
    end
end


function mean_prediction_error(y, ŷ)
    n = length(y)
    if n != length(ŷ)
        throw(DimensionMismatch("Arrays y and ŷ must have the same length."))
    end
    mpe = sum(sum(y .- ŷ)) / n
    return mpe
end



# Define the RNN architecture
mutable struct ODE_RNN
    rnn::Flux.Recur
    f_theta::Chain
    output::Chain
end

Flux.@functor ODE_RNN

function ODE_RNN(input_size::Int, hidden_size::Int, output_size::Int)
    cell = Flux.GRUCell(input_size + hidden_size, hidden_size)
    rnn = Flux.Recur(cell)
    f_theta = Chain(Dense(hidden_size, hidden_size, tanh))
    output = Chain(Dense(hidden_size, output_size))
    ODE_RNN(rnn, f_theta, output)
end

function (m::ODE_RNN)(timestamps::Vector{Float32}, datapoints::Matrix{Float32})
    h = Vector{Matrix{Float32}}(undef, length(timestamps))
    h[1] = m.rnn(vcat(zeros(Float32, size(m.rnn.cell.state0)...), datapoints[:,1]))

    dummy_p = Float32[]

    for i in 2:length(timestamps)
        tspan = (timestamps[i-1], timestamps[i])
        function ode_system(du, u, p, t)
            du .= m.f_theta(u)
        end
        prob = ODEProblem(ode_system, h[i-1], tspan)
        h_prime = solve(prob, Tsit5(), save_everystep=false, reltol=1e-7, abstol=1e-9)(tspan[2])
        rnn_input = vcat(h_prime, datapoints[:,i])
        h[i] = m.rnn(rnn_input)
    end

    return hcat(m.output.(h)...)
end




function agent(environment, hyperParams::HyperParameter)
    println("Hello people I am here")

    gym = pyimport("gym")
    if environment == "LunarLander-v2"
        global env = gym.make(environment, continuous=true)
    else
        global env = gym.make(environment)
    end

    envParams = EnvParameter()

    # Reset Parameters
    ## ActionenvP
    envParams.action_size =        env.action_space.shape[1]
    envParams.action_bound =       env.action_space.high[1]
    envParams.action_bound_high =  env.action_space.high
    envParams.action_bound_low =   env.action_space.low
    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low

    
    global fθ = ODE_RNN(envParams.state_size + envParams.action_size, 10, envParams.state_size)
    global opt = Flux.setup(Flux.Optimise.Adam(0.01), fθ)

    buffer = ReplayBuffer(hyperParams.buffer_size)

    episode = 0
    
    for i in 1:hyperParams.training_episodes
        
        frames = 0
        s, info = env.reset()
        episode_rewards = 0
        t = false
        
        while true

            #a = action(μθ, s, true, envParams, hyperParams)
            a = env.action_space.sample()
            s´, r, terminated, truncated, _ = env.step(a)
            
            terminated | truncated ? t = true : t = false
            
            episode_rewards += r

            remember(buffer, s, a, r, s´, t)

            if episode > hyperParams.train_start

                S, A, R, S´, T = sample(buffer, hyperParams.batch_size)


                dθ = Flux.gradient(m -> Flux.Losses.mse(m(collect(1.f0:hyperParams.batch_size), vcat(S, A)), S´), fθ)
                #Flux.update!(opt, fθ, dθ[1])

            end

            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end

        end


        push!(hyperParams.episode_steps, frames)
        push!(hyperParams.episode_reward, episode_rewards)
        
        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(hyperParams.critic_loss[end]) | Actor Loss: $(hyperParams.actor_loss[end]) | Steps: $(frames)")
        episode += 1

    end


        
    
    #end
    
    return hyperParams
    
end


