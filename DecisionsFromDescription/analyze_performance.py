import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches
import seaborn as sns
import math
from scipy import stats

plt.rcParams["figure.figsize"] = (3.46327,2.1)

def softmax(x, temp):
    return np.exp(temp*x) / np.sum(np.exp(temp*x), axis=-1, keepdims=True)

# plot gpt3
df = pd.read_csv("data/text-davinci-002/experiment_choice13k.csv")

random_value = (0.5 * df.valueA + 0.5 * df.valueB)
optimal_value = df[['valueA', 'valueB']].to_numpy().max(-1)
regret_random = (optimal_value - random_value)
print(regret_random.mean())

actionA = (df.action == 'F').to_numpy()
regret_davinci = np.concatenate([optimal_value[actionA], optimal_value[~actionA]]) - np.concatenate([df.valueA[actionA], df.valueB[~actionA]])
print(regret_davinci.mean())

print(regret_davinci.shape)
result = stats.ttest_ind(regret_davinci, regret_random, equal_var=True, alternative='less')
print(regret_davinci.shape[0] + regret_random.shape[0]- 2)
print(result.statistic)
print(result.pvalue)
print()

c13k_fp = "data/c13k_selections.csv"
c13k = pd.read_csv(c13k_fp)
c13k_problems = pd.read_json("data/c13k_problems.json", orient='index')
c13k_w_gambles = c13k.join(c13k_problems, how="left")

valuesA = []
valuesB = []
for index, row in c13k_w_gambles.iterrows():
    value_A = 0
    for item_A in row.A:
        value_A += item_A[1] * item_A[0]

    value_B = 0
    for item_B in row.B:
        value_B += item_B[1] * item_B[0]

    valuesA.append(value_A)
    valuesB.append(value_B)

c13k_w_gambles['valueA'] = valuesA
c13k_w_gambles['valueB'] = valuesB

human_optimal_value = c13k_w_gambles[['valueA', 'valueB']].to_numpy().max(-1)
print('here')
human_value = (c13k_w_gambles['valueB'] * c13k_w_gambles['bRate'] + c13k_w_gambles['valueA'] * (1 - c13k_w_gambles['bRate']))
regret_human = human_optimal_value - human_value
print(regret_human.mean())

human_std = 0

df = pd.read_csv("data/text-ada-001/experiment_choice13k.csv")
actionA = (df.action == 'F').to_numpy()
optimal_value = df[['valueA', 'valueB']].to_numpy().max(-1)
regret_ada = np.concatenate([optimal_value[actionA], optimal_value[~actionA]]) - np.concatenate([df.valueA[actionA], df.valueB[~actionA]])
print(regret_ada.mean())

result = stats.ttest_ind(regret_ada, regret_random, equal_var=True, alternative='less')
print(regret_ada.shape[0] + regret_random.shape[0]- 2)
print(result.statistic)
print(result.pvalue)
print()

df = pd.read_csv("data/text-babbage-001/experiment_choice13k.csv")
actionA = (df.action == 'F').to_numpy()
optimal_value = df[['valueA', 'valueB']].to_numpy().max(-1)
regret_babbage = np.concatenate([optimal_value[actionA], optimal_value[~actionA]]) - np.concatenate([df.valueA[actionA], df.valueB[~actionA]])
print(regret_babbage.mean())
result = stats.ttest_ind(regret_babbage, regret_random, equal_var=True, alternative='less')
print(regret_babbage.shape[0] + regret_random.shape[0]- 2)
print(result.statistic)
print(result.pvalue)
print()

df = pd.read_csv("data/text-curie-001/experiment_choice13k.csv")
actionA = (df.action == 'F').to_numpy()
optimal_value = df[['valueA', 'valueB']].to_numpy().max(-1)
regret_curie = np.concatenate([optimal_value[actionA], optimal_value[~actionA]]) - np.concatenate([df.valueA[actionA], df.valueB[~actionA]])
print(regret_curie.mean())
result = stats.ttest_ind(regret_curie, regret_random, equal_var=True, alternative='less')
print(regret_curie.shape[0] + regret_random.shape[0]- 2)
print(result.statistic)
print(result.pvalue)
print()

print(regret_human.shape)
result = stats.ttest_ind(regret_human, regret_davinci, equal_var=True, alternative='less')
print(regret_davinci.shape[0] + regret_human.shape[0]- 2)
print(result.statistic)
print(result.pvalue)
print()

values = np.array([regret_ada.mean(), regret_babbage.mean(), regret_curie.mean(), regret_davinci.mean(), regret_human.mean()])
plt.bar(np.arange(5), values, yerr=[regret_ada.std() / math.sqrt(len(regret_ada)), regret_babbage.std() / math.sqrt(len(regret_babbage)), regret_curie.std() / math.sqrt(len(regret_curie)), regret_davinci.std() / math.sqrt(len(regret_davinci)), regret_human.std() / math.sqrt(len(regret_human))], color=[ 'C0', 'C0', 'C0', 'C0', 'C1'], alpha=0.7, capsize=1, ecolor='dimgrey')
sns.despine()
plt.ylabel('Mean regret')
plt.xticks(np.arange(5), ['Ada', 'Babbage', 'Curie', 'Davinci', 'Humans'], rotation='vertical')
plt.tight_layout()
plt.axhline(y=regret_random.mean(), color='C3', linestyle='--', alpha=0.7)
plt.text(3.75, regret_random.mean() - 0.175, 'random', color='C3', alpha=0.7, size=8)
plt.savefig('figures/performance.pdf', bbox_inches='tight')
plt.show()
