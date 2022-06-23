import openai
import gym
import envs.bandits
import time
import pandas as pd
import numpy as np

num2words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
env = gym.make('wilson2014horizon-v0')
engine = "text-davinci-002"

def act(text):
    openai.api_key = "YOURKEY"
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = 1,
        temperature = 0.0,
    )
    return response.choices[0].text.strip()

def reset(actions, rewards):
    print(actions.shape)
    print(rewards.shape)
    played_machines = []
    observed_rewards = []
    for t in range(4):
        played_machines.append('J' if actions[t, 0] else 'F')
        observed_rewards.append(str(int(rewards[t, 0, 1].item())) if actions[t, 0] else str(int(rewards[t, 0, 0].item())))

    trials_left = 'one additional round' if rewards.shape[0] == 5 else 'six additional rounds'

    instructions = "You are going to a casino that owns two slot machines.\n"\
        "You earn money each time you play on one of these machines.\n\n"\
        "You have received the following amount of dollars when playing in the past: \n"

    history = "- Machine " + played_machines[0] + " delivered " + observed_rewards[0] + " dollars.\n"\
        "- Machine " + played_machines[1] + " delivered " + observed_rewards[1] + " dollars.\n"\
        "- Machine " + played_machines[2] + " delivered " + observed_rewards[2] + " dollars.\n"\
        "- Machine " + played_machines[3] + " delivered " + observed_rewards[3] + " dollars.\n"

    trials_left = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"
    question = "Q: Which machine do you choose?\n"\
        "A: Machine"

    return instructions, history, trials_left, question

def step(history, action, rewards, t):
    trials_left = env.rewards.shape[0] - 5 - t

    if trials_left > 1:
        trials_left = num2words[trials_left] + " additional rounds"
    else:
        trials_left = num2words[trials_left] + " additional round"

    trials_left_string = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"

    if action == "F":
        trial_reward = int(rewards[t + 4, 0, 0].item())
    elif action == "J":
        trial_reward = int(rewards[t + 4, 0, 1].item())
    else:
        return None

    history += "- Machine " + action + " delivered " + str(trial_reward) + " dollars.\n"

    return history, trials_left_string

num_tasks = 320 * 10

for task in range(num_tasks):
    actions = [None, None, None, None]
    env.reset()

    instructions, history, trials_left, question = reset(env.action, env.rewards)

    for t in range(env.rewards.shape[0] - 4):
        prompt = instructions + history + "\n" + trials_left + question
        print(prompt)
        action = act(prompt)
        if action == 'F':
            action_to_append = 0
        elif action == 'J':
            action_to_append = 1
        else:
            action_to_append = None
        actions.append(action_to_append)
        print(action)
        history, trials_left = step(history, action, env.rewards, t)
        if prompt is None:
            break
        print('')

    print(actions)
    if prompt is not None:
        data = []
        for trial in range(env.rewards.shape[0]):
            action = actions[trial] if trial >= 4 else env.action[trial, 0].item()
            row = [task, trial, env.mean_reward[0, 0].item(), env.mean_reward[0, 1].item(), env.rewards[trial, 0, 0].item(),  env.rewards[trial, 0, 1].item(), action]
            data.append(row)
        df = pd.DataFrame(data, columns=['task', 'trial', 'mean0', 'mean1', 'reward0', 'reward1', 'choice'])
        print(df)
        df.to_csv('data/' + engine + '/experiment' + str(task) + '.csv')
