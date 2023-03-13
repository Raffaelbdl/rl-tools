import gymnasium as gym
import numpy as np

from rl_tools.agent import Agent
from rl_tools.saver import Saver


def train_single_env(config: dict, env: gym.Env, agent: Agent):
    if not isinstance(env, gym.Env):
        raise TypeError(
            "envs is not a gymnasium environment, please use env_type='gymnasium'"
        )

    logs = {"steps": 0, "episodes": 0}
    saver = Saver(config["name"])

    n_env_steps = config["n_env_steps"]
    save_frequency = config["save_frequency"]

    observation, info = env.reset()
    for step in range(n_env_steps):
        logs["steps"] = step

        action, log_prob = agent.get_action(np.expand_dims(observation, 0))
        next_observation, reward, done, trunc, info = env.step(action)

        agent.buffer.add(observation, action, log_prob, reward, done, next_observation)

        if agent.improve_condition:
            logs = logs | agent.improve()

        next_observation = observation

        if done or trunc:
            observation, _ = env.reset()
            logs["episodes"] += 1

        if save_frequency > 0 and step % save_frequency == 0:
            saver.save(f"save_at_{step}", agent.params)


def train_envpool(config: dict, envs: gym.Env, agent: Agent):
    if not isinstance(envs, gym.Env):
        raise TypeError(
            "envs is not a gymnasium environment, please use env_type='gymnasium'"
        )

    logs = {"steps": 0}
    saver = Saver(config["name"])

    n_envs = config["n_envs"]
    n_env_steps = config["n_env_steps"]
    save_frequency = config["save_frequency"]

    observations, infos = envs.reset()
    for step in range(n_env_steps // n_envs):
        logs["steps"] = step * n_envs

        actions, log_probs = agent.get_action(observations)
        next_observations, rewards, dones, truncs, infos = envs.step(actions)

        agent.buffer.add(
            observations, actions, log_probs, rewards, dones, next_observations
        )

        if agent.improve_condition:
            logs = logs | agent.improve()

        next_observations = observations

        if save_frequency > 0 and step % save_frequency == 0:
            saver.save(f"save_at_{step}", agent.params)
