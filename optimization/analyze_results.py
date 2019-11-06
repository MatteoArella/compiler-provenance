from joblib import load
import json
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.json import json_normalize
from matplotlib.ticker import PercentFormatter

data = load('../train_dataset.joblib')

# plot classes count
counts = data['opt'].value_counts(normalize=True).apply(lambda x: x*100)
ax = counts.plot(kind='bar', rot=0)
plt.ylim(0, 100)
for i in ax.patches:
    ax.text(i.get_x()+.08, i.get_height()/2, '%.2f%%' % i.get_height(), fontsize=16, color='white')

plt.xlabel('Classes')
plt.ylabel('Classes count (%)')
plt.savefig('images/classes-count.png', bbox_inches='tight', dpi=300)

