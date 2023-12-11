"""
Simulation: takes bandit intervention stategies and a bandit generative model and runs a simulation.

Inspriation from: https://github.com/iosband/ts_tutorial/blob/master/src/base/experiment.py
"""
import numpy as np
from typing import Optional, Union

from src.bandit_data import  BanditData

class IpwPrimaryOutcome:

    def __int__(self,
                agent,
                bandit_env,
                n_obs: int,
                num_steps : int,
                burn_in : Optional[bool]

                ) -> None:

        self.bandit_env = bandit_env
        self.agent = agent
        self.n_obs
        self.num_steps = num_steps
        self.true_reward = np.zeros(self.num_steps)
        self.expected_reward = np.zeros(self.num_steps)
        self.current_step = 0
        self.burn_in = burn_in

        if self.burn_in:
            self.bandit = bandit_env.gen_sample(self.n_obs)

        return None

    def __str__(self):
        return f"IpwPrimaryOutcome(agent = {str(self.agent)!r})"

    def run_step(self):

        x = self.bandit_env.get_context(num_samples=n)
        actions_eps, propensities_eps = eps_greedy.pick_action(x, epsilon=.5)




