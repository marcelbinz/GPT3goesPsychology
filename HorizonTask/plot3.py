import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


#files = glob.glob("data/data3005/*")
files = glob.glob("data/text-davinci-002/*")

mode = 'equal'

plt.rcParams["figure.figsize"] = (2,2)


horizon1_reward_differences = []
horizon1_choices = []
horizon6_reward_differences = []
horizon6_choices = []

for file in files:
    df = pd.read_csv(file)

    horizon = 1 if len(df) == 5 else 6

    choices_B = df[df.trial < 4].choice.sum()
    if mode == 'equal':
        reward_difference = df.mean1[0] - df.mean0[0]
        choice = int(df[df.trial == 4].choice)
        if choices_B == 2:
            if horizon == 1:
                horizon1_reward_differences.append(reward_difference)
                horizon1_choices.append(choice)
            else:
                horizon6_reward_differences.append(reward_difference)
                horizon6_choices.append(choice)
    else:
        if choices_B == 1: #case: x3 A
            reward_difference = df.mean1[0] - df.mean0[0]
            choice = int(df[df.trial == 4].choice)
            if horizon == 1:
                horizon1_reward_differences.append(reward_difference)
                horizon1_choices.append(choice)
            else:
                horizon6_reward_differences.append(reward_difference)
                horizon6_choices.append(choice)
        if choices_B == 3: #case: x3 B
            reward_difference = df.mean0[0] - df.mean1[0]
            choice = 1 - int(df[df.trial == 4].choice)
            if horizon == 1:
                horizon1_reward_differences.append(reward_difference)
                horizon1_choices.append(choice)
            else:
                horizon6_reward_differences.append(reward_difference)
                horizon6_choices.append(choice)

print(np.array(horizon1_choices).mean())
print(np.array(horizon6_choices).mean())
print(np.array(horizon1_choices).shape)
log_reg_eh1 = sm.Logit(np.array(horizon1_choices), np.stack((np.array(horizon1_reward_differences), np.ones(len(horizon1_reward_differences))), axis=-1)).fit()
log_reg_eh6 = sm.Logit(np.array(horizon6_choices), np.stack((np.array(horizon6_reward_differences), np.ones(len(horizon6_reward_differences))), axis=-1)).fit()


x = np.stack((np.linspace(-30, 30, 1000), np.ones(1000)), axis=-1)

y_eh1 = log_reg_eh1.predict(x)
print(y_eh1.shape)
y_eh6 = log_reg_eh6.predict(x)



plt.plot(x[:, 0], y_eh1)
plt.plot(x[:, 0], y_eh6)

plt.legend(["Horizon 1", 'Horizon 6'], frameon=False, bbox_to_anchor=(-0.2,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=2, handlelength=0.5, handletextpad=0.5)
sns.despine()
plt.xlabel('Mean reward difference')
if mode == 'equal':
    plt.ylabel('p(right)')
else:
    plt.ylabel('p(more informative)')
plt.ylim(0, 1)
plt.xlim(-30, 30)
plt.savefig('choice_' + mode + '.pdf', bbox_inches='tight')
plt.show()
