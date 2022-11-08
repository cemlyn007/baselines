from typing import Callable, Sequence

import gym
import numpy as np
from .vec_env import VecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]):
        """
        Arguments:

        env_fns: iterable of callables functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_terminateds = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_truncateds = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec
        self.render_mode = self.envs[0].render_mode

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            (obs,
             self.buf_rews[e],
             self.buf_terminateds[e],
             self.buf_truncateds[e],
             self.buf_infos[e]) = self.envs[e].step(action)
            if self.buf_terminateds[e] or self.buf_truncateds[e]:
                obs, _ = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(),
                np.copy(self.buf_rews),
                np.copy(self.buf_terminateds),
                np.copy(self.buf_truncateds),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs, self.buf_infos[e] = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf(), self.buf_infos.copy()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        if self.envs[0].render_mode != 'rgb_array':
            raise ValueError('Render mode must be rgb_array.')
        # else...
        return [env.render() for env in self.envs]

    def render(self):
        if self.num_envs == 1:
            return self.envs[0].render()
        else:
            return super().render()
