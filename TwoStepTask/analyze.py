import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

sums = np.zeros(4)
counts = np.zeros(4)

for run in range(93):
    df = pd.read_csv('data/text-davinci-002/experiment_' + str(run) + '.csv')

    stay = (df.action1.to_numpy()[1:] == df.action1.to_numpy()[:-1]).astype(float) # == df.action1.to_numpy()
    reward = df.reward[:-1]
    common = (df.action1.to_numpy()[:-1] == df.state.to_numpy()[:-1])

    sums[0] += stay[reward & common].sum()
    sums[1] += stay[reward & ~common].sum()
    sums[2] += stay[(1 - reward) & common].sum()
    sums[3] += stay[(1 - reward) & ~common].sum()

    counts[0] += len(stay[reward & common])
    counts[1] += len(stay[reward & ~common])
    counts[2] += len(stay[(1 - reward) & common])
    counts[3] += len(stay[(1 - reward) & ~common])

print(sums/counts)

plt.bar(np.arange(4), sums/counts, color=['C0', 'C1', 'C0', 'C1'], alpha=0.7)
plt.ylim(0.5, 1)

import matplotlib.pyplot as plt

blue_patch = mpatches.Patch(color='C0', label='common')
orange_patch = mpatches.Patch(color='C1', label='rare')
plt.legend(handles=[blue_patch, orange_patch],frameon=False)
sns.despine()
plt.ylabel('stay probability')
plt.xticks([0.5, 2.5], ['rewarded', 'unrewarded'])
plt.show()
