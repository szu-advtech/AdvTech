import warnings
from scipy import stats, integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# warnings.filterwarnings('ignore')

# data = pd.read_csv('parameters3.dat', header=None, encoding='utf-8', delimiter="\ ", quoting=csv.QUOTE_NONE)
data = pd.read_table('parameters0.dat', sep='\ ', encoding='utf-8')

str = "EUrdistance_280w+18Bor"
df = data[str]

sum = 0
for i in df:
    sum += i
print(sum/2710)

# print(df)
# df.plot.kde()
# df.plot.density()
# plt.plot([0, 0], [0, 0.2], color="black")
df.plot(kind="hist", bins=50, color="gray", density=True)
df.plot(kind="kde", color="red")
# plt.xlim((-2.5, 2.5))
# my_x_ticks = np.arange(-2, 2, 2)
# plt.xticks(my_x_ticks)
# plt.yticks([])
plt.title(str)
plt.show()
