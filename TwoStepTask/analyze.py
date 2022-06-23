import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats
import math

plt.rcParams["figure.figsize"] = (3.46327,2.46327)

sums = np.zeros(4)
counts = np.zeros(4)

stays = [[], [], [], []]
all_rewards = []

for run in range(200):
    df = pd.read_csv('data/text-davinci-002/experiment_' + str(run) + '.csv')

    stay = (df.action1.to_numpy()[1:] == df.action1.to_numpy()[:-1]).astype(float)
    reward = df.reward[:-1]
    common = (df.action1.to_numpy()[:-1] == df.state.to_numpy()[:-1])

    all_rewards.append(reward.to_numpy())
    print(stay[reward & common].shape)
    sums[0] += stay[reward & common].sum()
    sums[1] += stay[reward & ~common].sum()
    sums[2] += stay[(1 - reward) & common].sum()
    sums[3] += stay[(1 - reward) & ~common].sum()

    stays[0].append(stay[reward & common])
    stays[1].append(stay[reward & ~common])
    stays[2].append(stay[(1 - reward) & common])
    stays[3].append(stay[(1 - reward) & ~common])


    counts[0] += len(stay[reward & common])
    counts[1] += len(stay[reward & ~common])
    counts[2] += len(stay[(1 - reward) & common])
    counts[3] += len(stay[(1 - reward) & ~common])

stays[0] = np.concatenate(stays[0])
stays[1] = np.concatenate(stays[1])
stays[2] = np.concatenate(stays[2])
stays[3] = np.concatenate(stays[3])

all_rewards = np.concatenate(all_rewards)

print("reward probability:")
print(all_rewards.mean())

print(stays[0].mean())
print(stays[1].mean())
print(stays[2].mean())
print(stays[3].mean())

print(stays[0].shape)
print(stays[1].shape)
print(stays[2].shape)
print(stays[3].shape)
print(sums/counts)

print(stays[1].shape[0] + stays[0].shape[0] - 2)
print(stats.ttest_ind(stays[1], stays[0], equal_var=True, alternative='less'))
print(stays[2].shape[0] + stays[3].shape[0] - 2)
print(stats.ttest_ind(stays[3], stays[2], equal_var=True, alternative='greater'))

print(stats.chi2_contingency([stays[1].sum(), stays[0].sum()]))

plt.bar(np.arange(4), sums/counts, color=['C0', 'C1', 'C0', 'C1'], alpha=0.7, yerr=[stays[0].std() / math.sqrt(len(stays[0])), stays[1].std() / math.sqrt(len(stays[1])), stays[2].std() / math.sqrt(len(stays[2])), stays[3].std() / math.sqrt(len(stays[3]))])
plt.ylim(0.5, 1)
blue_patch = mpatches.Patch(color='C0', label='common')
orange_patch = mpatches.Patch(color='C1', label='rare')
plt.legend(handles=[blue_patch, orange_patch], frameon=False, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=2, handlelength=1.5, handletextpad=0.5, mode="expand")
sns.despine()
plt.ylabel('stay probability')
plt.xticks([0.5, 2.5], ['rewarded', 'unrewarded'])
plt.tight_layout()
plt.savefig('figures/tst.pdf', bbox_inches='tight')
plt.show()
