import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (4,2.5)

df = pd.read_csv("data/text-davinci-002/experiment.csv")
probA = np.zeros(17)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

for t in range(1, df.task.max() + 1):
    df_task = df[df.task == t]
    log_probs = np.array([df_task.logprobA, df_task.logprobB])
    probs = softmax(log_probs)
    probA[t-1] = probs[0]

df = pd.read_csv("data/text-davinci-002/experiment_reverse.csv")
probA_reverse = np.zeros(17)

for t in range(1, df.task.max() + 1):
    df_task = df[df.task == t]
    log_probs = np.array([df_task.logprobA, df_task.logprobB])
    probs = softmax(log_probs)
    probA_reverse[t-1] = probs[0]

prob_A = (probA + probA_reverse) / 2

original_probA = np.array([0.18, 0.83, 0.20, 0.65, 0.14, 0.73, 0.92, 0.42, 0.92, 0.30, 0.22, 0.16, 0.69, 0.18, 0.70, 0.72, 0.17])

print(((probA > 0.5) == (original_probA > 0.5)).sum())

xticks = [
    "1 versus 2",
    "3 versus 4",
    "7 versus 8",
    "3 versus 7",
    "4 versus 8",
    "5 versus 9",
    "6 versus 10",
    "16 versus 17",
    "4 versus 11",
    "5 versus 6",
    "9 versus 10",
    "12 versus 13",
    "14 versus 15",
]

original_log_odds = np.log2(np.array([
    original_probA[0]/original_probA[1],
    original_probA[2]/original_probA[3],
    original_probA[6]/original_probA[7],
    original_probA[2]/original_probA[6],
    original_probA[3]/original_probA[7],
    original_probA[4]/original_probA[8],
    original_probA[5]/original_probA[9],
    original_probA[15]/original_probA[16],
    original_probA[3]/original_probA[10],
    original_probA[4]/original_probA[5],
    original_probA[8]/original_probA[9],
    original_probA[11]/original_probA[12],
    original_probA[13]/original_probA[14],
    ]))

gpt3_log_odds = np.log2(np.array([
    probA[0]/probA[1],
    probA[2]/probA[3],
    probA[6]/probA[7],
    probA[2]/probA[6],
    probA[3]/probA[7],
    probA[4]/probA[8],
    probA[5]/probA[9],
    probA[15]/probA[16],
    probA[3]/probA[10],
    probA[4]/probA[5],
    probA[8]/probA[9],
    probA[11]/probA[12],
    probA[13]/probA[14],
    ]))

print(original_log_odds)

plt.scatter(np.arange(1, 14), original_log_odds)
plt.scatter(np.arange(1, 14), gpt3_log_odds)
plt.xticks(np.arange(1, 14), xticks, rotation='vertical')
plt.ylabel('log(odds ratio)')
plt.legend(['human', 'text-davinci-002'], frameon=False, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=3, handlelength=0.5, handletextpad=0.5, mode='expand')
plt.ylim(-2.9, 2.9)
sns.despine()
plt.tight_layout()
plt.axhline(y=0.0, color='grey', linestyle='--')
plt.savefig('ptcontrast.pdf', bbox_inches='tight')
plt.show()
