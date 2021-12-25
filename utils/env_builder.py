import gym


class EnvBuilder:
    def __init__(self, env_name, **kwargs) -> None:
        self.env_name = env_name
        self.kwargs = kwargs
        
    def __call__(self) -> gym.Env:
        return gym.make(self.env_name, **self.kwargs)