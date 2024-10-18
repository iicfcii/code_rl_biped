

from gymnasium.envs.registration import register

register(
    id="biped",
    entry_point="my_envs.biped:BipedEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0
)
