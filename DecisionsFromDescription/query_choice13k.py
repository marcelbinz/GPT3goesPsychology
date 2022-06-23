import numpy as np
import pandas as pd
import openai

engine = "text-davinci-002"

def act(text):
    openai.api_key = "YOURKEY"
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = 1,
        temperature = 0.0,
        logprobs=2,
    )
    return response.choices[0].text.strip(), response.choices[0].logprobs.top_logprobs[0]

c13k_problems = pd.read_json("data/c13k_problems.json", orient='index')

print(len(c13k_problems))
random_problems = np.arange(len(c13k_problems))
data = []
for t, index in enumerate(random_problems):

    value_A = 0
    text_A = "- Option F: "
    for item_A in c13k_problems.iloc[index].A:
        value_A += item_A[1] * item_A[0]
        text_A += str(item_A[1]) + " dollars with " + str(round(item_A[0] * 100, 4)) + "% chance, "
    text_A = text_A[:-2]
    text_A += ".\n"

    value_B = 0
    text_B = "- Option J: "
    for item_B in c13k_problems.iloc[index].B:
        value_B += item_B[1] * item_B[0]
        text_B += str(item_B[1]) + " dollars with " + str(round(item_B[0] * 100, 4)) + "% chance, "
    text_B = text_B[:-2]
    text_B += ".\n"

    text = "Q: Which option do you prefer?\n\n"
    if np.random.choice([True, False]):
        text += text_A
        text += text_B
    else:
        text += text_B
        text += text_A

    text += "\nA: Option"

    action, log_probs = act(text)

    # fix if answer is not in top 2
    if not (" F" in log_probs):
        log_probs[" F"] = -9999
    if not (" J" in log_probs):
        log_probs[" J"] = -9999


    row = [index, value_A, value_B, action, log_probs[" F"], log_probs[" J"]]
    data.append(row)
    print(text)
    print(action)
    print()
    if ((t % 500) == 0):
        df = pd.DataFrame(data, columns=['task', 'valueA', 'valueB', 'action', 'logprobA', 'logprobB'])
        print(df)
        df.to_csv('data/' + engine + '/experiment_choice13k.csv')

df = pd.DataFrame(data, columns=['task', 'valueA', 'valueB', 'action', 'logprobA', 'logprobB'])
print(df)
df.to_csv('data/' + engine + '/experiment_choice13k.csv')
