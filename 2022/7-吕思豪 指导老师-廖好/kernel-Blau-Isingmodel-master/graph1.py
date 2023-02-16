import warnings
from scipy import stats, integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # warnings.filterwarnings('ignore')
#
# # data = pd.read_csv('parameters3.dat', header=None, encoding='utf-8', delimiter="\ ", quoting=csv.QUOTE_NONE)
# data = pd.read_table('ME16-12_sociodemographics.dat', sep='\ ', encoding='utf-8')
#
# str = "3ME12"
# df = data[str]
#
# print(df)
# # df.plot.kde()
# # df.plot.density()
# # plt.plot([0, 0], [0, 1], color="black")
# df.plot(kind="hist", bins=20, color="gray", density=True)
# df.plot(kind="kde", color="red")
# # plt.xlim((-2.5, 2.5))
# # my_x_ticks = np.arange(-2, 2, 2)
# # plt.xticks(my_x_ticks)
# # plt.yticks([])
# plt.title(str)
# plt.show()


area = {}

area['E05000064'] = []
area['E05000064'].append(['ME16', 'Me12'])
vec = []
vec.append(['4education'])  # 4education
vec.append(['5age'])  # 5age
vec.append(['6gender'])  # 6gender
vec.append(['7distance', '8distance'])  # 7 8 distance
vec.append(['9income'])  # 9income
area['E05000064'].append(vec)
area['E05000064'].append('1BoroughID')  # 1BoroughID

print(area.keys())
print()
print(area)
print()
print()
print(area['E05000064'][0][0])