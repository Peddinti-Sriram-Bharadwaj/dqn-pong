#!/usr/bin/env python3
import gymnasium as gym
import time
import argparse
import numpy as np
import torch
import cv2
import ale_py

# Register Atari environments
gym.register_envs(ale_py)

from lib import wrappers
from lib import dqn_model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME, help="Environment name")
    args = parser.parse_args()

    # 1. Setup Environment
    try:
        env = wrappers.make_env(args.env, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"Error: Could not find environment '{args.env}'.")
        print("Try running: pip install \"gymnasium[atari, accept-rom-license]\"")
        exit(1)

    # 2. Load Network
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    print(f"Loading model from {args.model}...")
    
    # Load on CPU
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)

    state, _ = env.reset()
    total_reward = 0.0

    print("---------------------------------------------")
    print(" WATCHING AGENT PLAY")
    print(" Press 'Q' to quit.")
    print("---------------------------------------------")

    while True:
        frame = env.render()
        
        if frame is not None:
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_big = cv2.resize(img_bgr, (600, 450), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Agent View", img_big)
        
        # --- THE FIX IS HERE ---
        # NumPy 2.0 Fix: Removed 'copy=False' so it can copy if needed
        state_v = torch.tensor(np.array([state]))
        # -----------------------

        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        if done or truncated:
            print(f"Game Finished! Total Reward: {total_reward}")
            state, _ = env.reset()
            total_reward = 0.0

    env.close()
    cv2.destroyAllWindows()
