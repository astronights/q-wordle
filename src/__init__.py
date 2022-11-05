from gym.envs.registration import register

register(
    id="QWordle-v0", entry_point="src.envs.qwordle:QWordle",
)

register(
    id="QWordle2-v0", entry_point="src.envs.qwordle2:QWordle2",
)

register(
    id="QWordle3-v0", entry_point="src.envs.qwordle3:QWordle3",
)