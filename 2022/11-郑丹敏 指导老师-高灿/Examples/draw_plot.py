import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xlrd import open_workbook

wb=open_workbook('../plot_result/ukbench AUS.xlsx')
tb=wb.sheets()[0]
data=[]
for r in range(1,tb.nrows):
    val=[]
    for c in range(tb.ncols):
        val.append(tb.cell_value(r,c))
    data.append(tuple(val))
print("data",data)

df = pd.DataFrame(data,
                  columns=pd.Index(["1VS", "NN", "TNN", "OSNN_CV", "OSNN"]),
                  index=['12','9','6','3'])
color=["lightgreen","khaki","coral","green","mediumpurple"]
ax = df.plot.bar(rot=0,color=color)
plt.legend(loc=(1,0.5))
# plt.ylabel("OSFM_μ")
# plt.ylabel("OSFM_M")
plt.ylabel("AUS")
plt.xlabel("The number of known classes")
# plt.title("caltech-256")
# 单独设置Y坐标轴上(水平方向)的网格线
ax.yaxis.grid(color='gray',
              linestyle='--',
              linewidth=1,
              alpha=0.3)
print(ax)
plt.savefig("../plot_result/pic/ukbench AUS.png",bbox_inches = 'tight')   # 加上这个bbox_inches = 'tight'图片就能完全显示了
plt.show()


