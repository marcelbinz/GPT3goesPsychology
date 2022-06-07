

import openai
import pandas as pd
import numpy as np

engine = "text-davinci-002"

def act(text):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = 1,
        temperature = 0.0,
        logprobs=2,
    )
    return response.choices[0].text.strip(), response.choices[0].logprobs.top_logprobs[0]

tasks = []

default_question = "Q: Which option do you prefer?\n\n"

task1 = default_question + "- Option F: 33% chance at 2,500 dollars, a 66% chance at 2,400 dollars, and a 1% chance of 0 dollars.\n"\
"- Option J: Guaranteed 2,400 dollars.\n" +  "\nA: Option"
tasks.append(task1)

task2 = default_question + "- Option F: 33% chance of 2,500 dollars (67% chance of 0 dollars).\n"\
"- Option J: 34% chance of 2,400 dollars (66% chance of 0 dollars).\n" +  "\nA: Option"
tasks.append(task2)

task3 = "- Option F: 80% chance of 4,000 dollars (20% chance of 0 dollars).\n"\
"- Option J: 100% guarantee of 3,000 dollars.\n" + "\nA: Option"
tasks.append(task3)

task4 = default_question + "- Option F: 20% chance of 4,000 dollars (80% chance of 0 dollars).\n"\
"- Option J: 25% chance of 3,000 dollars (75% chance of 0 dollars).\n" +  "\nA: Option"
tasks.append(task4)

task5 = default_question +"- Option F: 45% chance of 6,000 dollars (55% chance of 0 dollars).\n"\
"- Option J: 90% chance of 3,000 dollars (10% chance of 0 dollars).\n" +  "\nA: Option"
tasks.append(task5)

task6 =  default_question + "- Option F: 0.1% chance of 6,000 dollars (99.9% chance of 0 dollars).\n"\
"- Option J: 0.2% chance of 3,000 dollars (99.8% chance of 0 dollars).\n" +"\nA: Option"
tasks.append(task6)

task7 = default_question + "- Option F: 80% chance of losing 4,000 dollars (20% chance of losing 0 dollars).\n"\
"- Option J: 100% guarantee of losing 3,000 dollars.\n" + "\nA: Option"
tasks.append(task7)

task8 = default_question + "- Option F: 20% chance of losing 4,000 dollars (80% chance of losing 0 dollars).\n"\
"- Option J: 25% chance of losing 3,000 dollars (75% chance of losing 0 dollars).\n" +  "\nA: Option"
tasks.append(task8)

task9 = default_question + "- Option F: 45% chance of losing 6,000 dollars (55% chance of losing 0 dollars).\n"\
"- Option J: 90% chance of losing 3,000 dollars (10% chance of losing 0 dollars).\n" +  "\nA: Option"
tasks.append(task9)

task10 = default_question + "- Option F: 0.1% chance of losing 6,000 dollars (99.9% chance of losing 0 dollars).\n"\
"- Option J: 0.2% chance of losing 3,000 dollars (99.8% chance of losing 0 dollars).\n" + "\nA: Option"
tasks.append(task10)

task11 =  "Imagine you are playing a game with two levels, but you have to make a choice about the second level before you know the outcome of the first. "\
"At the first level, there is a 75% chance that the game will end without you winning anything, and a 25% chance that you will advance to the second level.\n\n"\
"Q: What would you choose in the second level?\n\n"\
"- Option F: 80% chance of 4,000 dollars (20% chance of 0 dollars).\n"\
"- Option J: 100% guarantee of 3,000 dollars.\n\nA: Option"
tasks.append(task11)

task12 = "Imagine we gave you 1,000 dollars right now to play a game.\n\n" + default_question + \
"- Option F: 50% chance to gain an additional 1,000 dollars (50% chance of gaining 0 dollars beyond what you already have).\n"\
"- Option J: 100% guarantee of gaining an additional 500 dollars.\n\nA: Option"
tasks.append(task12)

task13 = "Imagine we gave you 2,000 dollars right now to play a game.\n\n" + default_question + \
"- Option F: 50% chance you will lose 1,000 dollars (50% chance of losing 0 dollars).\n"\
"- Option J: 100% chance you will lose 500 dollars.\n\nA: Option"
tasks.append(task13)

task14 =  default_question + "- Option F: 25% chance of 6,000 dollars (75% chance of 0 dollars).\n"\
"- Option J: 25% chance of 4,000 dollars (25% chance of 2,000 dollars, 50% chance of 0 dollars).\n" + "\nA: Option"
tasks.append(task14)

task15 = default_question + "- Option F: 25% chance of losing 6,000 dollars (75% chance of losing nothing).\n"\
"- Option J: 25% chance of losing 4,000 dollars (25% chance of 2,000 dollars, 50% chance of 0 dollars).\n" +  "\nA: Option"
tasks.append(task15)

task16 = default_question + "- Option F: 0.1% chance at 5,000 dollars (99.9% chance of 0 dollars).\n"\
"- Option J: 100% guarantee of 5 dollars.\n" +  "\nA: Option"
tasks.append(task16)

task17 =  default_question + "- Option F: 0.1% chance of losing 5,000 dollars (99.9% chance of losing nothing).\n"\
"- Option J: 100% guarantee of losing 5 dollars.\n" + "\nA: Option"
tasks.append(task17)

data = []
for i, task in enumerate(tasks):
    print(task)
    action, log_probs = act(task)
    #action, log_probs = np.random.choice(['F', 'J']), {" F": 0.0, " J": 0.0}
    row = [i+1, action, log_probs[" F"], log_probs[" J"]]
    print(row)
    data.append(row)
df = pd.DataFrame(data, columns=['task', 'action', 'logprobA', 'logprobB'])
print(df)
df.to_csv('data/' + engine + '/experiment.csv')
