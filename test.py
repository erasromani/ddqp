import os
import gym
import torch
import time
import numpy as np
import random
import math
import torch.nn.functional as F
import torchvision.transforms as T
import json

from torch import optim
from torch import nn
from model import ActorNet
from utils import Transition, ReplayMemory, FrameHistory
from torch.nn.utils import clip_grad_norm_

# ---- HYPERPARAMETERS ---- #
N_EPISODES = 15
HISTORY_LENGTH = 4

# ---- SETTINGS ---- #
RENDER = True
ACTION_SPACE_SIZE = (4,)
STATE_SPACE_SIZE = (24,)


def select_action(actor, state):
    """
    This method selects an action based on the state per the policy network.
    """
    actor.eval()
    with torch.no_grad():
        action = actor(torch.stack(state.history)[None, ...].to(device))
    return action.cpu().detach().squeeze().clamp(-1, 1).numpy()


def update_state(frame, state):
    """
    This method updates the state based on a new frame given by one step of the gym environment.
    """
    state.push(transform_frame(frame))
    while len(state) < state.history_length:
        state.push(transform_frame(frame))


def run_episode(env, agent, state, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    frame = env.reset()
    while True:
        
        # get state history
        update_state(frame, state)
        
        # perform inference
        action = select_action(agent, state)

        next_frame, reward, done, info = env.step(action)
        episode_reward += reward     
        frame = next_frame
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


def transform_frame(frame):
    """
    This method transforms frames outputed by one step of the gym environment to a grayscale tensor.
    """
    # return torch.from_numpy(frame).float()
    return (torch.from_numpy(frame).float() - 0.2599063813686371) / 0.5198184847831726


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "BipedalWalkerHardcore-v3"
    env = gym.make(env_name).unwrapped
        
    model_dir = "models/"
    model_fn = "20201124-080538-checkpoint-4000.pth"
    results_dir = "./results/"

    if not os.path.exists(results_dir):
        os.mkdir(results_dir) 

    cs_cnn = [16, 32, 64, 64]
    agent  = ActorNet(cs_cnn, size=STATE_SPACE_SIZE, c_in=HISTORY_LENGTH, c_out=ACTION_SPACE_SIZE[0]).to(device)                       # input is the state
    st = torch.load('/'.join([model_dir, model_fn]), map_location=device)['actor']
    agent.load_state_dict(st)
    for param in agent.parameters(): param.requires_grad_(False)
    agent.eval()

    state = FrameHistory(HISTORY_LENGTH)
    episode_rewards = []

    # run episodes
    for i in range(N_EPISODES):
        episode_reward = run_episode(env, agent, state, rendering=RENDER)
        print("[EPISODE]: %i, [REWARD]: %i" % (i, episode_reward))
        episode_rewards.append(episode_reward)

    # save results
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    

    fname = "results/results_ddpg_agent-%s.json" % "".join(model_fn.split(".")[:-1])
    
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    fh.close()
    print('... finished')
