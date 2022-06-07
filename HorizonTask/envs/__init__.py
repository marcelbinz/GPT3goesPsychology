from gym.envs.registration import register

register(
    id='wilson2014horizon-v0',
    entry_point='envs.bandits:HorizonTaskWilson',
    kwargs={'num_actions': 2, 'reward_scaling': 1, 'reward_std': 8, 'num_forced_choice': 4},
)
