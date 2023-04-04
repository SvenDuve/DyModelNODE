module DyModelNODE


using Conda
using DifferentialEquations
using Flux
using Flux.Optimise
using Parameters
using PyCall
using SciMLSensitivity
import StatsBase
using Zygote




export HyperParameter, 
        modelEnv,
        ReplayBuffer,
        remember,
        sample,
        setReward,
        ODE_RNN,
        train_step!




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
    batch_size::Int =                       8
    trajectory::Int =                       20
    # Metrics
    episode_reward::Array{Float32} =        []
    model_loss::Array{Float32} =            []
    reward_loss::Array{Float32} =           []
    episode_steps::Array{Int} =             []
    # Discount
    γ::Float32 =                            0.99f0
    # Learning Rates
    model_η::Float64 =                      0.001
    hidden::Int =                           20
    reward_η::Float64 =                     0.001
    # Agents
    store_frequency::Int =                  10
    trained_model =                        []
    trained_reward =                        []
    # Mean Predition Error
    mpe =                                   []
    accuracy =                              []
    tolerance::Float64 =                    0.05
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

    # State progression verified manually with small batch_size
    
    start = StatsBase.sample(1:(size(buffer.memory)[1] - batch_size))
    batch = buffer.memory[start:(start+batch_size-1)]
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


function setReward(state_size, action_size)

    return Chain(Dense(state_size + action_size, 64, tanh),
                    Dense(64, 64, tanh),
                    Dense(64, 1))
                    
end



mutable struct ODE_RNN
    cell::Flux.RNNCell
    hidden::Int
    f_theta::Chain
    output::Chain
end


Flux.@functor ODE_RNN


function ODE_RNN(input_size::Int, hidden_size::Int, output_size::Int)
    cell = Flux.RNNCell(input_size, hidden_size)
    hidden = hidden_size
    f_theta = Chain(Dense(hidden_size, hidden_size, tanh))#, Dense(100,hidden_size, tanh))
    output = Chain(Dense(hidden_size, output_size))
    ODE_RNN(cell, hidden, f_theta, output)
end


function (m::ODE_RNN)(timestamps, datapoints)

    h = Zygote.Buffer(zeros32(m.hidden, size(timestamps)[1]+1))

    for (i, el) in enumerate(timestamps)

        tspan = (el - 1.f0, el)
        f(u, p, t) = m.f_theta(u)
        prob = ODEProblem(f, h[:,i], tspan)
        sol = solve(prob, AutoTsit5(Rosenbrock23()), abstol=1e-8,reltol=1e-8, save_everystep=false)#, Tsit5(), reltol=1e-8, abstol=1e-8)
        if sol.retcode != :Success
            @show sol.retcode
        end
        h[:,i+1] = m.cell(sol[2], datapoints[:,i])[1]

    end

    y = m.output(h[:,2:end])

    return y
end


function accuracy(y_true, y_pred, tolerance)
    correct = sum(abs.(y_true .- y_pred) .<= tolerance)
    return correct / length(y_true)
end


function train_step!(S, A, R, S´, T, fθ, Rϕ, ep::EnvParameter, hp::HyperParameter)

    X = vcat(S, A)
    timestamps = Float32[i for i in 1:size(X)[2]]
    
    # Train both critic networks

    sum(T[1:(end-1)]) > 0 && return

    try

        dθ = Flux.gradient(m -> Flux.Losses.mse(m(timestamps, X), S´), fθ)
        Flux.update!(model_opt, fθ, dθ[1])
        
        dϕ = Flux.gradient(m -> Flux.Losses.mse(m(X), hcat(R...)), Rϕ)
        Flux.update!(reward_opt, Rϕ, dϕ[1])

    catch

        println("Solve most probably divergent, this only affects the current hidden state solve, but has no effect on prior or further solves")

    end

end



function modelEnv(environment, hyperParams::HyperParameter)

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

    global fθ = ODE_RNN(envParams.state_size + envParams.action_size, hyperParams.hidden, envParams.state_size)
    global model_opt = Flux.setup(Flux.Optimise.Adam(hyperParams.model_η), fθ)

    global Rϕ = setReward(envParams.state_size, envParams.action_size)
    global reward_opt = Flux.setup(Flux.Optimise.Adam(hyperParams.reward_η), Rϕ)

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

                for j in 1:hyperParams.batch_size

#                    S, A, R, S´, T = sample(buffer, StatsBase.sample(2:hyperParams.trajectory))
                    S, A, R, S´, T = sample(buffer, hyperParams.trajectory)
                    train_step!(S, A, R, S´, T, fθ, Rϕ, envParams, hyperParams)


                end
            end

            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end

        end

        losses = []
        reward_losses = []
        acc = []
        
        for l in 1:10
            
            S, A, R, S´, T = sample(buffer, hyperParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            X = vcat(S, A)
            timestamps = Float32[i for i in 1:size(X)[2]]
            Ŝ = fθ(timestamps, X)
            
            push!(losses, Flux.Losses.mse(Ŝ, S´))
            push!(reward_losses, Flux.Losses.mse(Rϕ(X), hcat(R...)))
            push!(acc, accuracy(S´, Ŝ, hyperParams.tolerance))

        end

        if episode % hyperParams.store_frequency == 0

            # S, A, R, S´, T = sample(buffer, hyperParams.trajectory)
            # if sum(T[1:(end-1)]) == 0
            #     X = vcat(S, A)
            #     timestamps = Float32[i for i in 1:size(X)[2]]
            #     Ŝ = fθ(timestamps, X)
            #     @show S
            #     @show A
            #     @show S´
            #     @show Ŝ
            # end

            push!(hyperParams.trained_model, deepcopy(fθ))
            push!(hyperParams.trained_reward, deepcopy(Rϕ))
        end

        push!(hyperParams.model_loss, StatsBase.mean(losses))
        push!(hyperParams.reward_loss, StatsBase.mean(reward_losses))
        push!(hyperParams.accuracy, StatsBase.mean(acc))
        push!(hyperParams.episode_steps, frames)
        push!(hyperParams.episode_reward, episode_rewards)
        
        println("Episode: $episode | Accuracy: $(round(hyperParams.accuracy[end], digits=2)) | Model Loss: $(hyperParams.model_loss[end]) | Reward Loss: $(hyperParams.reward_loss[end])|  Steps: $(frames)")
        episode += 1

    end
    
    return hyperParams
    
end


end # module DyModelNODE


# This Works

# hp = agent("Pendulum-v1", HyperParameter(training_episodes=100, train_start=20, batch_size = 8, tolerance= 0.1))
# hp = agent("LunarLander-v2", HyperParameter(training_episodes=100, train_start=20, batch_size = 8, tolerance= 0.1))