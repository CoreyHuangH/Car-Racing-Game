import gymnasium as gym
from gymnasium.spaces import Discrete


class DiscreteCarRacing(gym.Wrapper):
    """
    Wrapper for the CarRacing environment that simplifies the action space to 5 discrete actions:
    0: No action
    1: Accelerate
    2: Turn left
    3: Turn right
    4: Brake
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(5)  # Simplified action space: 5 discrete actions
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # Convert discrete action to continuous action
        if action == 0:
            action = [0, 0, 0]  # No action
        elif action == 1:
            action = [0, 1, 0]  # Accelerate
        elif action == 2:
            action = [1, 0, 0]  # Turn left
        elif action == 3:
            action = [-1, 0, 0]  # Turn right
        elif action == 4:
            action = [0, 0, 0.8]  # Brake
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        return self.env.step(action)
