import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams["figure.figsize"] = (6.92654,2.4)

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
print("Probs A:")
print(prob_A)

#original_probA = np.array([0.18, 0.83, 0.20, 0.65, 0.14, 0.73, 0.92, 0.42, 0.92, 0.30, 0.22, 0.16, 0.69, 0.18, 0.70, 0.72, 0.17]) # original probs
original_probA = np.array([0.261, 0.619, 0.147, 0.525, 0.131, 0.658, 0.792, 0.514, 0.792, 0.442, 0.181, 0.267, 0.625, 0.206, 0.622, 0.581, 0.444]) # replication probs

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

original_log_odds = np.log(np.array([
    original_probA[1]/original_probA[0],
    original_probA[3]/original_probA[2],
    original_probA[6]/original_probA[7],
    original_probA[6]/original_probA[2],
    original_probA[3]/original_probA[7],
    original_probA[8]/original_probA[4],
    original_probA[5]/original_probA[9],
    original_probA[15]/original_probA[16],
    original_probA[3]/original_probA[10],
    original_probA[5]/original_probA[4],
    original_probA[8]/original_probA[9],
    original_probA[12]/original_probA[11],
    original_probA[14]/original_probA[15],
    ]))

gpt3_log_odds = np.log(np.array([
    probA[1]/probA[0],
    probA[3]/probA[2],
    probA[6]/probA[7],
    probA[6]/probA[2],
    probA[3]/probA[7],
    probA[8]/probA[4],
    probA[5]/probA[9],
    probA[15]/probA[16],
    probA[3]/probA[10],
    probA[5]/probA[4],
    probA[8]/probA[9],
    probA[12]/probA[11],
    probA[14]/probA[15],
    ]))

print(original_log_odds)

plt.scatter(np.arange(1, 14), original_log_odds, color=['C0', 'C0', 'C0', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C3', 'C3', 'C4', 'C5'], marker="o")
plt.scatter(np.arange(1, 14), gpt3_log_odds, color=['C0', 'C0', 'C0', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C3', 'C3', 'C4', 'C5'],marker="^")
plt.xticks(np.arange(1, 14), np.arange(1, 14))
plt.ylabel('log(odds ratio)')
plt.xlabel('Contrast')

custom_lines = [Line2D([0], [0], color='black', marker='o', linestyle='None'),
    Line2D([0], [0], color='black', marker='^', linestyle='None'),
    Line2D([0], [0], color='C0', marker='s', linestyle='None'),
    Line2D([0], [0], color='C1', marker='s', linestyle='None'),
    Line2D([0], [0], color='C2', marker='s', linestyle='None'),
    Line2D([0], [0], color='C3', marker='s', linestyle='None'),
    Line2D([0], [0], color='C4', marker='s', linestyle='None'),
    Line2D([0], [0], color='C5', marker='s', linestyle='None')]

plt.legend(custom_lines, ['Humans', 'GPT-3', 'Certainty effect', 'Reflection effect', 'Isolation effect', 'Overweighting bias', 'Framing effect', 'Magnitude perception'], frameon=False, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.ylim(-3.5, 3.5)
sns.despine()
plt.text(13.7, 0.4, 'bias present', color='grey', alpha=0.7, size=8, rotation='vertical')
plt.xlim(0, 14.4)
plt.arrow(14.25, 0.3, 0, 2.8, length_includes_head=True,
          head_width=0.2, head_length=0.2, color='grey', alpha=0.7)
plt.tight_layout()
plt.axhline(y=0.0, color='grey', linestyle='--')
plt.savefig('figures/contrast.pdf', bbox_inches='tight')
plt.show()
