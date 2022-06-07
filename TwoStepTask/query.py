import numpy as np
import pandas as pd
from scipy.special import softmax
import openai

def act(text):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = 1,
        temperature = 1,
    )
    return response.choices[0].text.strip()

def transition(action):
    if action == "X":
        return np.random.choice(['X', 'Y'], p=[0.7, 0.3])
    else:
        return np.random.choice(['X', 'Y'], p=[0.3, 0.7])

action_to_index = {"D": 0, "F": 1, "J": 2, "K": 3}

num_runs = 100
num_trials = 20
engine = "text-davinci-002"

for run in range(num_runs):
    reward_probs = np.random.uniform(0.25, 0.75, (4,))
    start_text = "You will travel to foreign planets in search of treasures.\n"\
    "When you visit a planet, you can choose an alien to trade with.\n"\
    "The chance of getting treasures from these aliens changes over time.\n"\
    "Your goal is to maximize the number of received treasures.\n\n"\

    previous_interactions = []
    data = []

    for i in range(num_trials):
        total_text = start_text
        if len(previous_interactions) > 0:
            total_text += "Your previous space travels went as follows:\n"
        for count, interaction in enumerate(previous_interactions):
            days = " day" if (len(previous_interactions) - count) == 1 else " days"
            total_text += "- " + str(len(previous_interactions) - count) + days + " ago, "
            total_text += interaction

        total_text += "\nQ: Do you want to take the spaceship to planet X or planet Y?\n"\
        "A: Planet"

        print(total_text)
        print()
        '''
        numerical_action1 = np.random.choice([0, 1]) # TODO
        action1 = 'X' if numerical_action1 == 0 else 'Y'
        '''
        action1 = act(total_text)

        total_text += " " + action1 + ".\n"
        state = transition(action1)
        numerical_state = 1 if state == 'X' else 2

        if state == "X":
            feedback = "You arrive at planet " + state + ".\n"\
            "Q: Do you want to trade with alien D or F?\n"\
            "A: Alien"
        elif state == "Y":
            feedback = "You arrive at planet " + state + ".\n"\
            "Q: Do you want to trade with alien J or K?\n"\
            "A: Alien"

        total_text += feedback

        print(total_text)
        print()

        '''
        numerical_action2 = np.random.choice([0, 1])

        if state == 'X':
            action2 = 'D' if numerical_action2 == 0 else 'F'
        else:
            action2 = 'J' if numerical_action2 == 0 else 'K'
        '''

        action2 = act(total_text)

        treasure = np.random.binomial(1, reward_probs[action_to_index[action2]], 1)[0]

        row = [run, i, action1, state, action2, treasure, reward_probs[0], reward_probs[1], reward_probs[2], reward_probs[3]]
        data.append(row)

        reward_probs += np.random.normal(0, 0.025, 4)
        reward_probs = np.clip(reward_probs, 0.25, 0.75)

        total_text += " " + action2 + ".\n"
        total_text += "You receive treasures." if treasure else "You receive junk."
        if treasure:
            feedback_item = "you boarded the spaceship to planet " + action1 + ", arrived at planet " + state + ", traded with alien " + action2 + ", and received treasures.\n"
        else:
            feedback_item = "you boarded the spaceship to planet " + action1 + ", arrived at planet " + state + ", traded with alien " + action2 + ", and received junk.\n"
        previous_interactions.append(feedback_item)

    df = pd.DataFrame(data, columns=['run', 'trial', 'action1', 'state', 'action2', 'reward', 'probsA', 'probsB', 'probsC', 'probsD'])
    df.to_csv('data/' + engine + '/experiment_' + str(run) + '.csv')
