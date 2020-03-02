import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filenmae = 'setup.csv'
data = pd.read_csv(filenmae)
data = data.sort_values(by=['MSE'], ascending=False)
width = 0.5
N = data.shape[0]
ind = np.arange(N)
MSEs = data['MSE']
labels = data['label']
plt.barh(ind, MSEs, width)
plt.yticks(ind, labels)
plt.xlabel('prediction error (mean square error)')
lossmessgae = "model1 : $k-\epsilon$ model \nmodel2 : $k-\omega$ model \nmodel3 : laminar"
plt.annotate(lossmessgae, xy=(0.55, 0.8), xycoords='axes fraction')
plt.tight_layout()
plt.savefig('MDTM_VS_SDTM')
