import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["figure.figsize"] = (3.46327,2.3)

conditions = ['common_cause', 'causal_chain']

for i, condition in enumerate(conditions):
    plt.figure(i)
    if condition == 'common_cause':
        ideal = [10, 10, 16, 4]
        human = [9.04, 6.26, 14.79, 3.29]
        gpt3 = [10, 10, 15, 10]
    elif condition == 'causal_chain':
        ideal = [16, 4, 16, 4]
        human = [14.04, 3.08, 13.67, 3.08]
        gpt3 = [10, 10, 15, 10]

    plt.bar(np.arange(4), ideal, width=0.25, alpha=0.7, color='C2')
    plt.bar(np.arange(4)+0.25, human, width=0.25, alpha=0.7, color='C1')
    plt.bar(np.arange(4)+0.5, gpt3, width=0.25, alpha=0.7, color='C0')
    plt.legend(['Ideal', 'Human', 'GPT-3'], frameon=False, bbox_to_anchor=(-0.0,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=3, handlelength=1.5, handletextpad=0.5, mode='expand')
    plt.xticks([0.25, 1.25, 2.25, 3.25], [r'$do(B = 1)$', r'$do(B = 0)$', r'$B = 1$', r'$B = 0$'])
    plt.ylabel('Reponse')
    plt.ylim(0, 16.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig("figures/" + condition + "_model.pdf", bbox_inches='tight')
    #plt.show()
