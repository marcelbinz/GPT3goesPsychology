import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams["figure.figsize"] = (3.46327,3.46327)

conditions = ['equal', 'unequal']

for i, mode in enumerate(conditions):
    plt.figure(i)

    files = glob.glob("data/text-davinci-002/e*")

    reward_differences = []
    horizons = []
    choices = []

    print(len(files))
    for file in files:
        df = pd.read_csv(file)
        horizon = 1 if len(df) == 10 else 0

        choices_B = df[df.trial < 4].choice.sum()
        if mode == 'equal':
            reward_difference = df.mean1[0] - df.mean0[0]
            choice = int(df[df.trial == 4].choice)
            if choices_B == 2:
                reward_differences.append(reward_difference)
                choices.append(choice)
                horizons.append(horizon)

        else:
            if choices_B == 1: #case: x3 A
                reward_difference = df.mean1[0] - df.mean0[0]
                choice = int(df[df.trial == 4].choice)

                reward_differences.append(reward_difference)
                choices.append(choice)
                horizons.append(horizon)

            if choices_B == 3: #case: x3 B
                reward_difference = df.mean0[0] - df.mean1[0]
                choice = 1 - int(df[df.trial == 4].choice)

                reward_differences.append(reward_difference)
                choices.append(choice)
                horizons.append(horizon)

    log_reg = sm.Logit(np.array(choices), np.stack((np.array(reward_differences), np.array(horizons), np.array(reward_differences) * np.array(horizons), np.ones(np.array(reward_differences).shape)), axis=-1)).fit()
    print(log_reg.summary())

    x_reward_differences = np.linspace(-30, 30, 1000)
    x_horizon6 = np.ones(1000)
    x_6 = np.stack((x_reward_differences, x_horizon6, x_horizon6 * x_reward_differences, np.ones(1000)), axis=-1)
    y_6 = log_reg.predict(x_6)

    x_reward_differences = np.linspace(-30, 30, 1000)
    x_horizon1 = np.zeros(1000)
    x_1 = np.stack((x_reward_differences, x_horizon1, x_horizon1 * x_reward_differences, np.ones(1000)), axis=-1)
    y_1 = log_reg.predict(x_1)

    plt.plot(x_1[:, 0], y_1, color='C0')
    plt.plot(x_6[:, 0], y_6, color='C0', ls='--')

    # plot human
    df = pd.read_csv('data/data.csv')
    if mode == 'equal':
        df = df[(df.Trial == 5) & (df.Info == 0)]
        reward_differences = (df.mu_R - df.mu_L).to_numpy().astype(float)
        choices = 1 - df.Choice.to_numpy()
        horizon = (df[(df.Trial == 5) & (df.Info == 0)].Horizon == 10).to_numpy().astype(float)
        interaction = horizon * reward_differences

        log_reg = sm.Logit(choices, np.stack((reward_differences, horizon, interaction, np.ones(reward_differences.shape)), axis=-1)).fit()
        print(log_reg.summary())
    else:
        # case: x3 A
        df_31 = df[(df.Trial == 5) & (df.Info == 1)]
        reward_differences_31 = (df_31.mu_R - df_31.mu_L).to_numpy().astype(float)
        choices_31 = 1 - df_31.Choice.to_numpy()
        horizons_31 = (df[(df.Trial == 5) & (df.Info == 1)].Horizon == 10).to_numpy().astype(float)

        # case: x3 B
        df_13 = df[(df.Trial == 5) & (df.Info == -1)]
        reward_differences_13 = (df_13.mu_L - df_13.mu_R).to_numpy().astype(float)
        choices_13 = df_13.Choice.to_numpy()
        horizons_13 = (df[(df.Trial == 5) & (df.Info == -1)].Horizon == 10).to_numpy().astype(float)

        choices = np.concatenate((choices_31, choices_13), axis=0)
        reward_differences = np.concatenate((reward_differences_31, reward_differences_13), axis=0)
        horizon = np.concatenate((horizons_31, horizons_13), axis=0)
        interaction = horizon * reward_differences

        log_reg = sm.Logit(choices, np.stack((reward_differences, horizon, interaction, np.ones(reward_differences.shape)), axis=-1)).fit()
        print(log_reg.summary())

    x_reward_differences = np.linspace(-30, 30, 1000)
    x_horizon6 = np.ones(1000)
    x_6 = np.stack((x_reward_differences, x_horizon6, x_horizon6 * x_reward_differences, np.ones(1000)), axis=-1)
    y_6 = log_reg.predict(x_6)

    x_reward_differences = np.linspace(-30, 30, 1000)
    x_horizon1 = np.zeros(1000)
    x_1 = np.stack((x_reward_differences, x_horizon1, x_horizon1 * x_reward_differences, np.ones(1000)), axis=-1)
    y_1 = log_reg.predict(x_1)

    plt.plot(x_1[:, 0], y_1, color='C1')
    plt.plot(x_6[:, 0], y_6, color='C1', ls='--')

    custom_lines = [Line2D([0], [0], color='black', linestyle='-'),
        Line2D([0], [0], color='C0', linestyle='-'),
        Line2D([0], [0], color='black', linestyle='--'),
        Line2D([0], [0], color='C1', linestyle='-')]

    plt.legend(custom_lines, ['Horizon 1','GPT-3',  'Horizon 6', 'Humans'], frameon=False, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=2, handlelength=1.5, handletextpad=0.5, mode='expand')
    sns.despine()
    plt.xlabel('Mean reward difference')
    if mode == 'equal':
        plt.ylabel('p(J)')
    else:
        plt.ylabel('p(more informative)')
    plt.ylim(0, 1)
    plt.xlim(-30, 30)
    plt.tight_layout()
    plt.savefig('figures/choice_' + mode + '.pdf', bbox_inches='tight')
