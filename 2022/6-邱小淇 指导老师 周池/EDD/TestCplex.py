"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

from docplex.mp.model import Model  #导出库，只用这一个就够了
model = Model() #创建模型
var_list = [i for i in range(0, 7)] #创建列表
X = model.binary_var_list(var_list, lb=0, name='X') #创建变量列表
#设定目标函数
model.maximize(11* X[0] + 9 * X[1] + 29 * X[2]+9* X[3]+21*X[4]+31*X[5]+22*X[6])
#添加约束条件
model.add_constraint(X[0]+X[1]+X[2] <= 2)
model.add_constraint(X[3] + X[4] >=1)
model.add_constraint(X[5] + X[6] >=1)
model.add_constraint(10* X[0] + 8* X[1] + 20 * X[2]+5* X[3]+13*X[4]+22*X[5]+10*X[6] <=60)
sol = model.solve() #求解模型
print(sol)  #打印结果

#
def get_solve(vnum,cnum,I,D,costy):
    Tcv_varlist=[(i,j) for i in range(cnum) for j in range(vnum)]

