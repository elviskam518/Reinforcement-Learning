

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import rldurham as rld
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("videos", exist_ok=True)


import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections, random



class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, buffer_limit=1000000):
        self.max_size = buffer_limit
        self.ptr = 0
        self.current_size = 0
        self.state = np.zeros((buffer_limit, obs_dim))
        self.action = np.zeros((buffer_limit, act_dim))
        self.next_state = np.zeros((buffer_limit, obs_dim))
        self.reward = np.zeros((buffer_limit, 1))
        self.not_done = np.zeros((buffer_limit, 1))

    def put(self, transition):
        s, a, r, s_prime, done = transition
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s_prime
        self.reward[self.ptr] = r
        self.not_done[self.ptr] = 1.0 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, n):
        ind = np.random.randint(0, self.current_size, size=n)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

    def size(self):
        return self.current_size



class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.fc4 = nn.Linear(obs_dim + act_dim, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class TD3Agent(nn.Module):
    def __init__(self,
                 env,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 buffer_limit=1000000,
                 gamma=0.99,
                 batch_size=256,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 exploration_noise=0.1):
        super().__init__()

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.total_it = 0
        self.total_steps = 0

        discrete_act, discrete_obs, act_dim, obs_dim = rld.env_info(env, print_out=True)
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(obs_dim, act_dim, buffer_limit=buffer_limit)

    def action(self, state, add_noise=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.act_dim)
            action = action + noise

        action = np.clip(action, -1.0, 1.0)
        return action

    def put(self, s, a, r, s_prime, done):
        self.memory.put((s, a, r, s_prime, done))

    def update(self):
        if self.memory.size() < self.batch_size:
            return
        self.total_it += 1
        state, action, reward, next_state, not_done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * not_done * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def episode(self):
        done = False
        s, _ = self.env.reset()
        info = {}
        total_reward = 0

        while not done:
            self.total_steps += 1

            if self.total_steps < 25000:
                a = np.random.uniform(-1, 1, size=self.act_dim)
            else:
                a = self.action(s, add_noise=True)

            s_prime, r, terminated, truncated, info = self.env.step(a)
            total_reward += r
            done = terminated or truncated

            self.put(s, a, r, s_prime, done)

            if self.total_steps >= 25000:
                self.update()
            s = s_prime

        info['total_reward'] = total_reward
        return info


env = rld.make("rldurham/Walker", render_mode="rgb_array", hardcore=True)
rld.seed_everything(42, env)

env = rld.Recorder(env, smoothing=10, video=True, logs=True,
                   video_folder="videos", video_prefix="cqst66-agent-video-hardcore")
recorder = env
recorder.video = False
tracker = rld.InfoTracker()

env = rld.transparent_wrapper(gym.wrappers.ClipReward)(env, min_reward=-10)

agent = TD3Agent(env, exploration_noise=0.1)



for n_epi in range(2000):
    if n_epi < 500:
        agent.exploration_noise = 0.1
    else:
        agent.exploration_noise = max(0.01, 0.1 - 0.09 * ((n_epi - 500) / 1500))

    recorder.video = (n_epi + 1) % 100 == 0 or n_epi == 1897
    info = agent.episode()
    env.add_stats(info, ignore_existing=True)
    tracker.track(info)
    print(f"Episode: {n_epi + 1}/2000, Score: {info['recorder']['r_sum']:.2f}", flush=True)

    if (n_epi + 1) % 100 == 0:
        tracker.plot(r_mean_=True, r_std_=True, r_sum=dict(linestyle=':', marker='x'))
        plt.savefig(f"graphs/progress_{n_epi + 1}.png")
        plt.close()

env.close()
env.write_log(folder="logs", file="cqst66-agent-log-hardcore.txt")
print("saved")