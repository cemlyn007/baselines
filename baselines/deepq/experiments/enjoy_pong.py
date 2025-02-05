import gym
from baselines import deepq


def main():
    env = gym.make("PongNoFrameskip-v4")
    env = deepq.wrap_atari_dqn(env)
    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        total_timesteps=0
    )

    while True:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_rew = 0
        while not (terminated or truncated):
            env.render()
            obs, rew, terminated, truncated, _ = env.step(model(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
