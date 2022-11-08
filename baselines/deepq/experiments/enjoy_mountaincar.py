import gym

from baselines import deepq
from baselines.common import models


def main():
    env = gym.make("MountainCar-v0")
    act = deepq.learn(
        env,
        network=models.mlp(num_layers=1, num_hidden=64),
        total_timesteps=0,
        load_path='mountaincar_model.pkl'
    )

    while True:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_rew = 0
        while not (terminated or truncated):
            env.render()
            obs, rew, terminated, truncated, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
