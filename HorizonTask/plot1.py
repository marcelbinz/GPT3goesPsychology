import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


regrets = np.zeros(7)
models = ["data/text-ada-001/*", "data/text-babbage-001/*", "data/text-curie-001/*", "data/text-davinci-002/*"]

plt.rcParams["figure.figsize"] = (2.25,2.25)


for i, model in enumerate(models):
    files = glob.glob(model)
    gpt3_regrets = []
    random_regrets = []
    greedy_regrets = []
    for file in files:
        df = pd.read_csv(file)
        if len(df) ==5:
            rewards_A = []
            rewards_B = []
            for t in range(4):
                if int(df[df.trial == t].choice):
                    rewards_B.append(df[df.trial == t].reward1)
                else:
                    rewards_A.append(df[df.trial == t].reward0)

            for t in range(4, df.trial.max() + 1):
                max_reward = np.max((df[df.trial == t].reward0, df[df.trial == t].reward1))
                random_reward = 0.5 * df[df.trial == t].reward0 + 0.5 * df[df.trial == t].reward1
                gpt_reward = df[df.trial == t].reward0 if int(df[df.trial == t].choice) == 0 else df[df.trial == t].reward1
                greedy_reward = df[df.trial == t].reward0 if np.array(rewards_A).mean() > np.array(rewards_B).mean() else df[df.trial == t].reward1

                if int(df[df.trial == t].choice):
                    rewards_B.append(df[df.trial == t].reward1)
                else:
                    rewards_A.append(df[df.trial == t].reward0)

                random_regrets.append(max_reward - random_reward)
                gpt3_regrets.append(max_reward - gpt_reward)
                greedy_regrets.append(max_reward - greedy_reward)

    regrets[i+1] = np.array(gpt3_regrets).mean()
    regrets[0] += np.array(random_regrets).mean()
    regrets[6] += np.array(greedy_regrets).mean()

regrets[0] = regrets[0] / len(models)
regrets[6] = regrets[6] / len(models)

df = pd.read_csv('data/data.csv')
df = df[df.Trial >= 5]
r_max = df[["mu_L", "mu_R"]].max(axis=1)
r_obs = df.Outcome
regrets[5] = (r_max - r_obs).mean()

print(regrets)
plt.bar(np.arange(7), regrets, color=['C0', 'C1', 'C1', 'C1', 'C1', 'C2', 'C0'], alpha=0.7)
sns.despine()
plt.ylabel('Mean regret')
plt.xticks(np.arange(7), ['random', 'ada-001', 'babbage-001', 'curie-001', 'davinci-002', 'human', 'greedy'], rotation='vertical')
plt.tight_layout()
plt.savefig('performance.pdf', bbox_inches='tight')
plt.show()

'''reward_difference = df.mean0[0] - df.reward1[0]
    horizon = 1 if len(df) == 5 else 6

    choice = int(df[df.trial == 4].choice)'''
