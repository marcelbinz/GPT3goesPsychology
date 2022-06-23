import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats
import math

plt.rcParams["figure.figsize"] = (3.46327,2.46327)

means = [0.796, 0.796, 0.648, 0.656]

plt.bar(np.arange(4), means, color=['C0', 'C1', 'C0', 'C1'], alpha=0.7)
plt.ylim(0.5, 1)

blue_patch = mpatches.Patch(color='C0', label='common')
orange_patch = mpatches.Patch(color='C1', label='rare')
plt.legend(handles=[blue_patch, orange_patch], frameon=False, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",  borderaxespad=0, ncol=2, handlelength=1.5, handletextpad=0.5, mode="expand")
sns.despine()
plt.ylabel('stay probability')
plt.xticks([0.5, 2.5], ['rewarded', 'unrewarded'])
plt.tight_layout()
plt.savefig('figures/tst_mf.pdf', bbox_inches='tight')
plt.show()
