"""Zdecentralizowany multi-agent dla stable-baselines3.

`MultiAgentVecEnv` bierze VecEnv z K środowiskami TubeAviary (każde z 2 dronami)
i wystawia go jako VecEnv z 2K niezależnymi agentami:

    obserwacja agenta = tylko lokalna obserwacja jego drona  (shape: (24,))
    akcja agenta      = komenda tylko dla jego drona          (shape: (4,))
    nagroda agenta    = jego własna nagroda (info['agent_rewards'][i])

PPO trenuje na tym jedną sieć, którą wykonują oba drony niezależnie
(parameter sharing, tzw. IPPO). Żaden dron nie widzi obserwacji drugiego
ani nie dostaje jego akcji; każdy wybiera akcję tylko z własnych danych.
"""
from copy import deepcopy

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class MultiAgentVecEnv(VecEnv):

    def __init__(self, venv, n_agents=2):
        self.venv = venv
        self.n_agents = n_agents
        obs_space, act_space = venv.observation_space, venv.action_space
        single_obs = spaces.Box(low=obs_space.low[0], high=obs_space.high[0], dtype=np.float32)
        single_act = spaces.Box(low=act_space.low[0], high=act_space.high[0], dtype=np.float32)
        super().__init__(venv.num_envs * n_agents, single_obs, single_act)

    #### Przekształcenia (K, n, ...) <-> (K*n, ...)
    def _split_obs(self, obs):
        return obs.reshape(self.num_envs, -1)

    def reset(self):
        return self._split_obs(self.venv.reset())

    def step_async(self, actions):
        self.venv.step_async(np.asarray(actions).reshape(self.venv.num_envs, self.n_agents, -1))

    def step_wait(self):
        obs, _, dones, infos = self.venv.step_wait()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        out_infos = []
        for k, info in enumerate(infos):
            ag_r = info.get("agent_rewards", np.zeros(self.n_agents))
            for i in range(self.n_agents):
                a_info = {key: v for key, v in info.items()
                          if key not in ("terminal_observation", "episode", "agent_rewards")}
                if "terminal_observation" in info:
                    a_info["terminal_observation"] = info["terminal_observation"][i]
                if "episode" in info:   # wynik zespołu z Monitora (potrzebny evaluate_policy dla każdego agenta)
                    a_info["episode"] = deepcopy(info["episode"])
                rewards[k * self.n_agents + i] = ag_r[i]
                out_infos.append(a_info)
        return self._split_obs(obs), rewards, np.repeat(dones, self.n_agents), out_infos

    def close(self):
        self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return self.venv.env_method(method_name, *args, **kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [all(self.venv.env_is_wrapped(wrapper_class))] * self.num_envs

    def seed(self, seed=None):
        return self.venv.seed(seed)
