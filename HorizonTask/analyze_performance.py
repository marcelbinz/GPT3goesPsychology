import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.lines import Line2D
from scipy import stats

plt.rcParams["figure.figsize"] = (3.46327,3.46327)

# human regret
df = pd.read_csv('data/data.csv')
df_h1 = df[(df.Trial == 5) & (df.Horizon==5)]
r_max = df_h1[["mu_L", "mu_R"]].max(axis=1)
r_obs = df_h1.Outcome
human_regrets_h1 = (r_max - r_obs).to_numpy()
human_regrets_h6 = []
for i in range(5, 11):
    df_h6 = df[(df.Trial == i) & (df.Horizon==10)]
    r_max = df_h6[["mu_L", "mu_R"]].max(axis=1)
    r_obs = df_h6.Outcome
    human_regrets_h6.append((r_max - r_obs).to_numpy())

human_regrets_h6 = np.array(human_regrets_h6).T
print(human_regrets_h6.shape)
print(human_regrets_h6.mean(0))

num_h1 = 0
num_h6 = 0
files = glob.glob("data/text-davinci-002/e*")

for file in files:
    df = pd.read_csv(file)
    if len(df) == 5:
        num_h1 += 1
    elif len(df) == 10:
        num_h6 += 1

regrets_h1 = np.zeros((num_h1, 1))
regrets_h6 = np.zeros((num_h6, 6))
random_regrets_h6 = np.zeros((num_h6, 6))

print(num_h1)
print(num_h6)

counter_h1 = 0
counter_h6 = 0

for file in files:
    df = pd.read_csv(file)
    for t in range(4, df.trial.max() + 1):
        max_reward = np.max((df[df.trial == t].mean0, df[df.trial == t].mean1))
        gpt_reward = df[df.trial == t].mean0 if int(df[df.trial == t].choice) == 0 else df[df.trial == t].mean1
        regret = (max_reward - gpt_reward)
        random_reward = 0.5 * df[df.trial == t].mean0 + 0.5 * df[df.trial == t].mean1
        random_regret = (max_reward - random_reward)
        #print(regret)
        if len(df) == 5:
            regrets_h1[counter_h1, t-4] = regret
            counter_h1 += 1
        elif len(df) == 10:
            regrets_h6[counter_h6, t-4] = regret
            random_regrets_h6[counter_h6, t-4] = random_regret
            if t == df.trial.max():
                counter_h6 += 1
print(regrets_h6.shape)
print(regrets_h6.mean(0))

custom_lines = [Line2D([0], [0], color='black', marker='s', linestyle='None'),
    Line2D([0], [0], color='C0', linestyle='-'),
    Line2D([0], [0], color='black', linestyle='--'),
    Line2D([0], [0], color='C1',  linestyle='-')]

print(regrets_h1.shape[0] + human_regrets_h1.shape[0] - 2 )
print(stats.ttest_ind(regrets_h1[:, 0], human_regrets_h1, equal_var=True))
print(regrets_h6.shape[0] + human_regrets_h6.shape[0] - 2 )
print(stats.ttest_ind(regrets_h6[:, 0], human_regrets_h6[:, 0], equal_var=True, alternative='less'))
print(human_regrets_h6.shape[0] + regrets_h6.shape[0] - 2 )
print(stats.ttest_ind(human_regrets_h6[:, -1], regrets_h6[:, -1], equal_var=True, alternative='less'))

regrets_full = np.concatenate((np.ravel(regrets_h1), np.ravel(regrets_h6)), axis=0)
human_regrets_full = np.concatenate((np.ravel(human_regrets_h1), np.ravel(human_regrets_h6)), axis=0)
print(regrets_full.mean())
print(regrets_full.std())

print(human_regrets_full.mean())
print(human_regrets_full.std())

print(regrets_full.shape)
print(human_regrets_full.shape)

print(stats.ttest_ind(regrets_full, human_regrets_full, equal_var=True, alternative='less'))


plt.axhline(y=random_regrets_h6.mean(), color='C3', linestyle='--', alpha=0.7)

plt.scatter(np.arange(1) + 1 -0.1, regrets_h1.mean(0), alpha=0.7,  marker='s', color='C0')
plt.errorbar(np.arange(1) + 1-0.1, regrets_h1.mean(0), alpha=0.7, yerr=(regrets_h1.mean(0) / math.sqrt(regrets_h1.shape[0])), color='C0')

plt.errorbar(np.arange(6) + 1-0.1, regrets_h6.mean(0), alpha=0.7, yerr=(regrets_h6.mean(0) / math.sqrt(regrets_h6.shape[0])), color='C0', linestyle='--',  marker='o')

plt.scatter(np.arange(1) + 1 +0.1, human_regrets_h1.mean(0), alpha=0.7, marker='s', color='C1')
plt.errorbar(np.arange(1) + 1 +0.1, human_regrets_h1.mean(0), alpha=0.7, yerr=(human_regrets_h1.mean(0) / math.sqrt(human_regrets_h1.shape[0])), color='C1')

plt.errorbar(np.arange(6) + 1 + 0.1, human_regrets_h6.mean(0), alpha=0.7, yerr=(human_regrets_h6.mean(0) / math.sqrt(human_regrets_h6.shape[0])), color='C1', linestyle='--', marker='o')

plt.text(5.375, random_regrets_h6.mean() - 0.35, 'random', color='C3', alpha=0.7, size=8)

sns.despine()
plt.ylabel('Mean regret')

plt.xlim(0.75, 6.25)
plt.xlabel('Trial')
plt.ylim(0, random_regrets_h6.mean() + 0.2)

plt.legend(custom_lines, ['Horizon 1','GPT-3',  'Horizon 6', 'Humans'], frameon=False, bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=2, handlelength=1.5, handletextpad=0.5, mode='expand')
plt.tight_layout()
plt.savefig('figures/regret.pdf', bbox_inches='tight')

plt.show()
