import openai
import numpy as np
from sklearn.utils import shuffle

text = "You have previously observed the following chemical substances in different wine casks:\n"

def act(text):
    openai.api_key = "YOURKEY"
    response = openai.Completion.create(
        engine = 'text-davinci-002',
        prompt = text,
        max_tokens = 100,
        temperature = 0.0,
        suffix = ' casks.'
    )
    return response.choices[0].text

data = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
])

print(data.shape)
data = shuffle(data)

for i in range(data.shape[0]):
    text += "- Cask " + str(i+1) + ": "
    if data[i, 0]:
        text += "substance A was present, "
    else:
        text += "substance A was absent, "
    if data[i, 1]:
        text += "substance B was present, "
    else:
        text += "substance B was absent, "
    if data[i, 2]:
        text += "substance C was present.\n"
    else:
        text += "substance C was absent.\n"


text += "\nYou have the following additional information from previous research:\n"\
        "- Substance A likely causes the production of substance B.\n"\
        "- Substance A likely causes the production of substance C.\n\n"


observation_text1 = text + "Imagine that you test 20 new casks in which you know that substance B is present. \nQ: How many of these new casks will contain substance C on average?\n"\
    "A:"

observation_text2 = text + "Imagine that you test 20 new casks in which you know that substance B is absent. \nQ: How many of these new casks will contain substance C on average?\n"\
    "A:"

intervention_text1 = text + "Imagine that you test 20 new casks in which you have manually added substance B. \nQ: How many of these new casks will contain substance C on average?\n"\
    "A:"

intervention_text2 = text + "Imagine that you test 20 new casks in which you have manually removed substance B. \nQ: How many of these new casks will contain substance C on average?\n"\
    "A:"

print(intervention_text1)


action = act(intervention_text1)
print(action)

action = act(intervention_text2)
print(action)

action = act(observation_text1)
print(action)

action = act(observation_text2)
print(action)
