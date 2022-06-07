import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


#files = glob.glob("data/data3005/*")
df = pd.read_csv('data/data.csv')

plt.rcParams["figure.figsize"] = (2,2)


mode = 'unequal'
if mode == 'equal':
    df_h1 = df[(df.Trial == 5) & (df.Info == 0) & (df.Horizon == 5)]
    df_h6 = df[(df.Trial == 5) & (df.Info == 0) & (df.Horizon == 10)]
    horizon1_reward_differences = (df_h1.mu_R - df_h1.mu_L).to_numpy().astype(float)
    horizon6_reward_differences = df_h6.mu_R - df_h6.mu_L.to_numpy().astype(float)
    horizon1_choices = 1 - df_h1.Choice.to_numpy()
    horizon6_choices = 1 - df_h6.Choice.to_numpy()
    print(horizon1_choices)

    log_reg_eh1 = sm.Logit(horizon1_choices, np.stack((horizon1_reward_differences, np.ones(len(horizon1_reward_differences))), axis=-1)).fit()
    log_reg_eh6 = sm.Logit(horizon6_choices, np.stack((horizon6_reward_differences, np.ones(len(horizon6_reward_differences))), axis=-1)).fit()

else:
    # case: x3 A
    df_h1_31 = df[(df.Trial == 5) & (df.Info == 1) & (df.Horizon == 5)]
    df_h6_31 = df[(df.Trial == 5) & (df.Info == 1) & (df.Horizon == 10)]
    h1_31_reward_differences = (df_h1_31.mu_R - df_h1_31.mu_L).to_numpy().astype(float)
    h6_31_reward_differences = df_h6_31.mu_R - df_h6_31.mu_L.to_numpy().astype(float)
    h1_31_choices = 1 - df_h1_31.Choice.to_numpy()
    h6_31_choices = 1 - df_h6_31.Choice.to_numpy()

    # case: x3 B
    df_h1_13 = df[(df.Trial == 5) & (df.Info == -1) & (df.Horizon == 5)]
    df_h6_13 = df[(df.Trial == 5) & (df.Info == -1) & (df.Horizon == 10)]
    h1_13_reward_differences = (df_h1_13.mu_L - df_h1_13.mu_R).to_numpy().astype(float)
    h6_13_reward_differences = df_h6_13.mu_L - df_h6_13.mu_R.to_numpy().astype(float)
    h1_13_choices = df_h1_13.Choice.to_numpy()
    h6_13_choices = df_h6_13.Choice.to_numpy()

    horizon1_choices = np.concatenate((h1_31_choices, h1_13_choices), axis=0)
    horizon6_choices = np.concatenate((h6_31_choices, h6_13_choices), axis=0)
    horizon1_reward_differences = np.concatenate((h1_31_reward_differences, h1_13_reward_differences), axis=0)
    horizon6_reward_differences = np.concatenate((h6_31_reward_differences, h6_13_reward_differences), axis=0)

    log_reg_eh1 = sm.Logit(horizon1_choices, np.stack((horizon1_reward_differences, np.ones(len(horizon1_reward_differences))), axis=-1)).fit()
    log_reg_eh6 = sm.Logit(horizon6_choices, np.stack((horizon6_reward_differences, np.ones(len(horizon6_reward_differences))), axis=-1)).fit()

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
plt.savefig('human_' + mode + '.pdf', bbox_inches='tight')
plt.show()
