from gym.envs.registration import register

register(
    id="QWordle-v0", entry_point="src.envs.qwordle:QWordle",
)