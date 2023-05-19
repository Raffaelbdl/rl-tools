try:
    import gymnasium as gym
    import numpy as np
except ImportError:
    print("The following packages are necessary to use the training loops")
    print("- gymnasium")
    print("- numpy")


from rltools.agent import Agent
from rltools.saver import Saver


def train_single_env(config: dict, env: gym.Env, agent: Agent):
    if not isinstance(env, gym.Env):
        raise TypeError(
            "envs is not a gymnasium environment, please use env_type='gymnasium'"
        )

    logs = {"steps": 0, "episodes": 0, "n_updates": 0, "episode_return": 0}
    saver = Saver(config["name"])

    n_env_steps = config["n_env_steps"]
    save_frequency = config["save_frequency"]

    observation, info = env.reset()
    episode_return = 0.0
    for step in range(n_env_steps):
        logs["steps"] = step

        action, log_prob = agent.get_action(np.expand_dims(observation, 0))
        next_observation, reward, done, trunc, info = env.step(action)

        episode_return += reward

        agent.buffer.add(observation, action, log_prob, reward, done, next_observation)

        if agent.improve_condition:
            logs = logs | agent.improve()
            logs["n_updates"] += (
                config["n_samples_and_updates"] * config["n_minibatches"]
            )

        observation = next_observation

        if done or trunc:
            observation, _ = env.reset()
            logs["episode_return"] = episode_return
            episode_return = 0.0
            print(logs["steps"], logs["episode_return"])

            logs["episodes"] += 1

        if save_frequency > 0 and step % save_frequency == 0:
            saver.save(f"save_at_{step}", agent.params)


def train_envpool(config: dict, envs: gym.Env, agent: Agent, use_wandb: bool = False):
    if not isinstance(envs, gym.Env):
        raise TypeError(
            "envs is not a gymnasium environment, please use env_type='gymnasium'"
        )

    if use_wandb:
        import wandb

    logs = {"steps": 0, "episodes": 0, "n_updates": 0, "episode_return": 0}
    saver = Saver(config["name"])

    n_envs = config["n_envs"]
    n_env_steps = config["n_env_steps"]
    save_frequency = config["save_frequency"]

    observations, infos = envs.reset()
    episode_returns = np.zeros((n_envs,))
    for step in range(n_env_steps // n_envs):
        logs["steps"] = step * n_envs

        actions, log_probs = agent.get_action(observations)
        next_observations, rewards, dones, truncs, infos = envs.step(actions)

        episode_returns += rewards

        agent.buffer.add(
            observations, actions, log_probs, rewards, dones, next_observations
        )

        if agent.improve_condition:
            logs = logs | agent.improve()
            logs["n_updates"] += (
                config["n_samples_and_updates"] * config["n_minibatches"]
            )

        observations = next_observations

        for i, done in enumerate(dones):
            if done:
                logs["episode_return"] = episode_returns[i]
                episode_returns[i] = 0.0
                print(logs["steps"], logs["episode_return"])

                logs["episodes"] += 1

                if use_wandb:
                    wandb.log(logs)

        if save_frequency > 0 and step % save_frequency == 0:
            saver.save(f"save_at_{step}", agent.params)


def train_envpool_with_value(
    config: dict, envs: gym.Env, agent: Agent, use_wandb: bool = False
):
    if not isinstance(envs, gym.Env):
        raise TypeError(
            "envs is not a gymnasium environment, please use env_type='gymnasium'"
        )

    if use_wandb:
        import wandb

    logs = {"steps": 0, "episodes": 0, "n_updates": 0, "episode_return": 0}
    saver = Saver(config["name"])

    n_envs = config["n_envs"]
    n_env_steps = config["n_env_steps"]
    save_frequency = config["save_frequency"]

    observations, infos = envs.reset()
    values = agent.get_value(observations)
    episode_returns = np.zeros((n_envs,))
    for step in range(n_env_steps // n_envs):
        logs["steps"] = step * n_envs

        actions, log_probs = agent.get_action(observations)
        next_observations, rewards, dones, truncs, infos = envs.step(actions)
        next_values = agent.get_value(next_observations)

        episode_returns += rewards

        agent.buffer.add(
            observations,
            actions,
            log_probs,
            rewards,
            dones,
            next_observations,
            values,
            next_values,
        )

        if agent.improve_condition:
            logs = logs | agent.improve()
            logs["n_updates"] += (
                config["n_samples_and_updates"] * config["n_minibatches"]
            )

        observations = next_observations
        values = next_values

        for i, done in enumerate(dones):
            if done:
                logs["episode_return"] = episode_returns[i]
                episode_returns[i] = 0.0
                print(logs["steps"], logs["episode_return"])

                logs["episodes"] += 1

                if use_wandb:
                    wandb.log(logs)

        if save_frequency > 0 and step % save_frequency == 0:
            saver.save(f"save_at_{step}", agent.params)
