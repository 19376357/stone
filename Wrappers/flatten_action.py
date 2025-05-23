import gym


class Flatten_Action(gym.ActionWrapper):
    '''
    Assuming actor model outputs tanh range [-1, 1]
    Convert the range [-1, 1] to [env.action_space.low, env.action_space.high]
    '''
    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high*action.
        :param action:
        :return: normalized action
        """
        action = gym.spaces.utils.unflatten(self.action_space, action)
        return action
