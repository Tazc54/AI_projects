import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Helper function to create an MLP
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# Main training function
def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):
    
    # Make environment, check spaces, get obs/act dims
    env = gym.make(env_name, render_mode="rgb_array")
    assert isinstance(env.observation_space, Box), "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Policy network
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # Helper functions
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        logits = logits_net(torch.as_tensor(obs, dtype=torch.float32))
        policy = Categorical(logits=logits)
        return policy.sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    # Training loop for one epoch
    def train_one_epoch():
        batch_obs, batch_acts, batch_weights = [], [], []
        batch_rets, batch_lens = [], []
        obs, _ = env.reset()
        done = False
        ep_rews = []
        finished_rendering_this_epoch = False

        while True:
            if render and not finished_rendering_this_epoch:
                env.render()

            # Save observation
            batch_obs.append(obs.copy())

            # Take action
            act = get_action(obs)
            obs, rew, done, _, _ = env.step(act)

            # Save action and reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_weights += [ep_ret] * ep_len
                obs, _ = env.reset()
                done, ep_rews = False, []

                if len(batch_obs) > batch_size:
                    break

        # Update policy
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # Training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(f'epoch: {i+1:3d} \t loss: {batch_loss:.3f} \t return: {np.mean(batch_rets):.3f} \t ep_len: {np.mean(batch_lens):.3f}')

    return logits_net, env, get_action

# Evaluate the trained policy and render as RGB array
def evaluate(env, get_action, max_steps=500):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    frames = []

    for _ in range(max_steps):
        frame = env.render()  # Render as RGB array
        frames.append(frame)  # Save the frame

        action = get_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

    env.close()
    return frames, total_reward

# Function to visualize the frames as a video
from matplotlib.animation import FuncAnimation

# Function to visualize the frames as a real-time animation
def display_frames_as_animation(frames, interval=50):
    """
    Displays a real-time animation of the frames.

    Args:
        frames (list): List of RGB arrays (frames).
        interval (int): Time interval between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(frames[0])
    ax.axis('off')

    def update(frame):
        img.set_array(frame)
        return [img]

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.show()


if __name__ == '__main__':
    # Train the policy
    trained_policy, trained_env, get_action = train(env_name='CartPole-v1', render=False)

    # Evaluate the trained policy
    print("\nEvaluating trained policy...")
    frames, total_reward = evaluate(trained_env, get_action)

    print(f"\nTotal Reward: {total_reward}")

    # Display the frames as a video
    display_frames_as_animation(frames)
