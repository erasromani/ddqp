import os
import gym
import torch
import time
import numpy as np
import random
import math
import torch.nn.functional as F
import torchvision.transforms as T

from itertools import count
from torch import optim
from torch import nn
from model import CriticNet, ActorNet
from utils import Transition, ReplayMemory, FrameHistory, exponential_moving_average
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


# ---- HYPERPARAMETERS ---- #
BATCH_SIZE = 256
GAMMA = 0.99
N_EPISODES = 5000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999996
BUFFER_SIZE = 1000000
MAX_ITERATIONS = 500
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
POLYAK_PARAM = 1e-3
HISTORY_LENGTH = 4

# ---- SETTINGS ---- #
RENDER = False
RECORD = True
ACTION_SPACE_SIZE = (4,)
STATE_SPACE_SIZE = (24,)
MODEL_SAVE_FREQUENCY = 100


def soft_update(target_model, local_model, tau):
    """
    This method updated the target network parameters by performing a polyak averaging between 
    the target network and the policy network.
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def update_epsilon(epsilon):
    """
    This method updates the epsilon parameter for epsilon greedy such that it decays as per EPS_START, 
    EPS_END, and EPS_DECAY_LENGTH.
    """
    if epsilon > EPSILON_END:
        epsilon *= EPSILON_DECAY
    return epsilon


def select_action(actor, state):
    """
    This method selects an action based on the state per the policy network.
    """
    actor.eval()
    with torch.no_grad():
        action = actor(torch.stack(state.history)[None, ...].to(device))
        noise = 2 * torch.rand_like(action) - 1
        action += epsilon * noise
    return action.cpu().detach().squeeze().clamp(-1, 1).numpy()


def optimize_actor_critic(replay_buffer, actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer):
    """
    This method optimizes the policy network by minimizing the TD error between the Q from the 
    policy network and the Q calculated through a Bellman backup via the target network.
    """
    global epsilon
    if len(replay_buffer) < BATCH_SIZE: return
    transitions = replay_buffer.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Manage edge cases 
    non_final_mask = torch.tensor(tuple(map(lambda x: x is not None,
                                          batch.next_state)), device=device, dtype=torch.bool).to(device)
    non_final_next_states = torch.stack([x for x in batch.next_state if x is not None]).to(device)
    
    # Create batch
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)

    # print(f"\n [STATE MEAN]: {state_batch.mean()}, [STATE STD]: {state_batch.std()}, [ACTION MEAN]: {action_batch.mean(dim=0)}, [ACTION STD]: {action_batch.std(dim=0)}\n")

    # ------------------------- update critic ------------------------------ #
    critic.train()
    next_state_q = torch.zeros(BATCH_SIZE, device=device)
    next_state_q[non_final_mask] = critic_target(non_final_next_states, actor_target(non_final_next_states)).squeeze()  # here we are getting the target q value of the next state
    expected_q_value = (next_state_q.unsqueeze(1) * GAMMA) + reward_batch                                               # value at terminal state is reward_batch

    critic_optimizer.zero_grad()
    q_value = critic(state_batch, action_batch)
    critic_loss = F.mse_loss(q_value, expected_q_value)
    critic_loss.backward()
    clip_grad_norm_(critic.parameters(), 100.0)
    critic_optimizer.step()

    # ------------------------- update actor ------------------------------ #
    actor.train()
    # critic.eval()
    actor_optimizer.zero_grad()
    actor_loss = -critic(state_batch, actor(state_batch)).mean()
    actor_loss.backward()
    clip_grad_norm_(actor.parameters(), 0.2)
    actor_optimizer.step()

    # Update the target network
    # polyak_averaging(actor_target, actor, POLYAK_PARAM)
    # polyak_averaging(critic_target, critic, POLYAK_PARAM)
    soft_update(actor_target, actor, POLYAK_PARAM)
    soft_update(critic_target, critic, POLYAK_PARAM)

    epsilon = update_epsilon(epsilon)
    
    # Record output
    if RECORD:
        actor_grad_norm = torch.stack([params.grad.data.norm() for params in actor.parameters()])
        critic_grad_norm = torch.stack([params.grad.data.norm() for params in critic.parameters()])
        writer.add_scalar('Actor Loss', actor_loss.item(), total_iterations)
        writer.add_scalar('Epsilon', epsilon, total_iterations)
        writer.add_scalar('Critic Loss', critic_loss.item(), total_iterations)
        writer.add_scalar('Min Critic Gradient Norm', critic_grad_norm.min().item(), total_iterations)
        writer.add_scalar('Max Critic Gradient Norm', critic_grad_norm.max().item(), total_iterations)
        writer.add_scalar('Min Actor Gradient Norm', actor_grad_norm.min().item(), total_iterations)
        writer.add_scalar('Max Actor Gradient Norm', actor_grad_norm.max().item(), total_iterations)


def transform_frame(frame):
    """
    This method transforms frames outputed by one step of the gym environment to a grayscale tensor.
    """
    # return torch.from_numpy(frame).float()
    return (torch.from_numpy(frame).float() - 0.2599063813686371) / 0.5198184847831726


def update_state(frame, state):
    """
    This method updates the state based on a new frame given by one step of the gym environment.
    """
    state.push(transform_frame(frame))
    while len(state) < state.history_length:
        state.push(transform_frame(frame))


def get_next_state(state, next_frame, done):
    """
    This method gets the next state given the next frame produces by one step of the gym environment.
    """
    if not done: 
        next_state = state.clone()
        next_state.push(transform_frame(next_frame))
    else:
        next_state = None
    return next_state


def store_transition(state, next_state, action, reward, replay_buffer):
    """
    This method stores the transition in the replay buffer.
    """
    replay_buffer.push(
            torch.stack(state.history),
            torch.from_numpy(action).float(),
            torch.stack(next_state.history) if next_state else next_state,
            torch.tensor([reward], dtype=torch.float32),
    )

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "BipedalWalkerHardcore-v3"
    env = gym.make(env_name).unwrapped
    model_dir = "./models/"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_fn = timestamp + "-checkpoint"
    log_dir = "./logs/"
    log_fn = timestamp + "-log.txt"
    tensorboard_dir = "./runs/"

    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir) 
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    cs_cnn = [16, 32, 64, 64]
    cs_fcn = [256, 64]
    actor  =         ActorNet(cs_cnn, size=STATE_SPACE_SIZE, c_in=HISTORY_LENGTH, c_out=ACTION_SPACE_SIZE[0]).to(device)                       # input is the state
    critic =        CriticNet(cs_cnn, cs_fcn, size_action=ACTION_SPACE_SIZE, size_state=(HISTORY_LENGTH, *STATE_SPACE_SIZE)).to(device)        # input is the concatenation of state and action
    actor_target  =  ActorNet(cs_cnn, size=STATE_SPACE_SIZE, c_in=HISTORY_LENGTH, c_out=ACTION_SPACE_SIZE[0]).to(device)                       # input is the state
    critic_target = CriticNet(cs_cnn, cs_fcn, size_action=ACTION_SPACE_SIZE, size_state=(HISTORY_LENGTH, *STATE_SPACE_SIZE)).to(device)        # input is the concatenation of state and action

    # actor_target.load_state_dict(actor.state_dict())
    # critic_target.load_state_dict(critic.state_dict())
    # for param in actor_target.parameters(): param.requires_grad_(False)
    # for param in critic_target.parameters(): param.requires_grad_(False)
    # actor_target.eval()
    # critic_target.eval()

    for param in actor_target.parameters(): param.requires_grad_(False)
    for param in critic_target.parameters(): param.requires_grad_(False)
    hard_update(actor_target, actor)
    hard_update(critic_target, critic)

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)

    replay_buffer = ReplayMemory(BUFFER_SIZE)
    state = FrameHistory(HISTORY_LENGTH)

    if RECORD: writer = SummaryWriter()
    f = open(log_dir + log_fn, "w")
    total_iterations = 0
    epsilon = EPSILON_START
    episode_rewards = []

    for episode in range(N_EPISODES):
        
        # Initialize the environment and states
        frame = env.reset()
        episode_reward = 0

        for iteration in count():

            # Execute iteration
            update_state(frame, state)                                                              # Update state
            action = select_action(actor, state)                                                    # Select action
            next_frame, reward, done, info = env.step(action)                                       # Perform action
            next_state = get_next_state(state, next_frame, done)                                    # Get next state
            store_transition(state, next_state, action, reward, replay_buffer)                      # Store transition in replay buffer
            optimize_actor_critic(                                                                  # Optimize actor critic
                    replay_buffer, 
                    actor, critic, 
                    actor_target, critic_target, 
                    actor_optimizer, critic_optimizer)

            # Increment episode reward
            episode_reward += reward
            if RECORD: writer.add_scalar('Reward', episode_reward, total_iterations)

            # Evaluate exit criteria
            if done or iteration > MAX_ITERATIONS: break
            
            # Set-up next iteration
            frame = next_frame
            total_iterations += 1
            if RENDER: env.render()

        # Set-up next episode
        episode_rewards.append(episode_reward)
        n = max(0, len(episode_rewards)-50)
        avg_rewards = sum(episode_rewards[n:]) / len(episode_rewards[n:])
        if RECORD and episode % MODEL_SAVE_FREQUENCY == 0:
            torch.save({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'actor_target': actor_target.state_dict(),
                    'critic_target': critic_target.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'episode': episode,
                }, '/'.join([model_dir, model_fn + "-" + str(episode) +".pth"]))
        f.write("[EPISODE]: %i, [ITERATION]: %i, [EPSILON]: %f,[EPISODE REWARD]: %i, [AVG REWARD PER 50 EPISODES]: %i, [ITERATION PER EPOCH]: %i\n" % (episode, total_iterations, epsilon, episode_reward, avg_rewards, iteration))
        env.close()
        
    f.write("\n[SUM OF REWARD FOR ALL %i EPISODES]: %i\n" % (N_EPISODES, sum(episode_rewards)))
    if RECORD: writer.close()
    f.close()
