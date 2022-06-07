import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (2.25,2.25)


regrets_h1 = np.zeros(1)
regrets_h6 = np.zeros(6)
h1_count = 0
h6_count = 0
files = glob.glob("data/text-davinci-002/*")

for file in files:
    df = pd.read_csv(file)
    if len(df) == 5:
        h1_count += 1
    else:
        h6_count += 1

    for t in range(4, df.trial.max() + 1):
        max_reward = np.max((df[df.trial == t].mean0, df[df.trial == t].mean1))
        gpt_reward = df[df.trial == t].mean0 if int(df[df.trial == t].choice) == 0 else df[df.trial == t].mean1
        regret = (max_reward - gpt_reward)
        print(regret)
        if len(df) == 5:
            regrets_h1[t-4] += regret
        else:
            regrets_h6[t-4] += regret
print(regrets_h1)
print(regrets_h6)

plt.plot(np.arange(1) + 1, regrets_h1 / h1_count, alpha=0.7)
plt.scatter(np.arange(1) + 1, regrets_h1 / h1_count, alpha=0.7)
plt.plot(np.arange(6) + 1, regrets_h6 / h6_count, alpha=0.7)
plt.scatter(np.arange(6) + 1, regrets_h6 / h6_count, alpha=0.7)
sns.despine()
plt.ylabel('Mean regret')

plt.xlim(0.5, 6.5)
plt.xlabel('Trial')
plt.ylim(0, 700/150)
plt.tight_layout()
plt.legend(["Horizon 1", 'Horizon 6'], frameon=False, bbox_to_anchor=(-0.2,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=2, handlelength=0.5, handletextpad=0.5)
plt.savefig('regretovertime.pdf', bbox_inches='tight')

plt.show()
