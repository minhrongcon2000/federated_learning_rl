from gym.envs.registration import register

register(id="SinglePath-v0", 
         entry_point="video_streaming_env.singlepath_env.envs.singlepath_gym:SinglepathEnvGym")
