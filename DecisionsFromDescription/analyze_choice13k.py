import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams["figure.figsize"] = (2,2)

def softmax(x, temp):
    return np.exp(temp*x) / np.sum(np.exp(temp*x), axis=-1, keepdims=True)

# plot gpt3
df = pd.read_csv("data/text-davinci-002/experiment_choice13k.csv")
print(df)

x = df.valueA-df.valueB

log_probs = np.array([df.logprobA, df.logprobB]).T
probs = softmax(log_probs, 1)
y = np.argmax(1-probs, -1)

log_reg = sm.Logit(y, np.stack((x, np.ones(len(x))), axis=-1)).fit()
x_pred = np.stack((np.linspace(-100, 100, 1000), np.ones(1000)), axis=-1)
y_pred = log_reg.predict(x_pred)

plt.plot(x_pred[:, 0], y_pred)

# plot human
c13k_fp = "data/c13k_selections.csv"
c13k = pd.read_csv(c13k_fp)
c13k_problems = pd.read_json("data/c13k_problems.json", orient='index')
c13k_w_gambles = c13k.join(c13k_problems, how="left")

x = []
y = []

for index, row in c13k_w_gambles.iterrows():
    value_A = 0
    for item_A in row.A:
        value_A += item_A[1] * item_A[0]

    value_B = 0
    for item_B in row.B:
        value_B += item_B[1] * item_B[0]
    print(value_A)
    print(value_B)
    x.append(value_A-value_B)
    y.append(1-row.bRate)

y = np.random.binomial(1, y)

log_reg = sm.Logit(y, np.stack((x, np.ones(len(x))), axis=-1)).fit()
x_pred = np.stack((np.linspace(-100, 100, 1000), np.ones(1000)), axis=-1)
y_pred = log_reg.predict(x_pred)

plt.plot(x_pred[:, 0], y_pred)

plt.legend(["human", 'text-davinci-002'], frameon=False, bbox_to_anchor=(-0.2,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=3, handlelength=0.5, handletextpad=0.5, labelspacing = 0)
sns.despine()
plt.ylabel('p(a)')
plt.xlabel('Reward difference')

plt.savefig('choice13k.pdf', bbox_inches='tight')

plt.show()
