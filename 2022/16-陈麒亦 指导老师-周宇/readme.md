## 实验环境搭建
- python (tested on version 3.8.13)
- pytorch (tested on version 1.11.0+cu115)
- torch-geometric (tested on version 2.1.0post1)
- numpy (tested on version 1.22.3)
- scikit-learn(tested on version 1.1.2)

## 文件夹:
- code：GraphCDA 的模型代码和训练代码。
- data：GraphCDA 所需的数据。
- datasets：几个公共数据库
- results：GraphCDA 运行的结果。
## 数据描述:
- d_d.csv: 疾病综合相似性
- c_c.csv: circRNA 综合相似性
- d_c.csv: 疾病-circRNA 关联表
- dss.csv: 疾病语义相似性
- cfs.csv: circRNA 功能相似性
- dgs.csv: 疾病高斯内核相似性
- cgs.csv: circRNA 高斯内核相似性
- disname.txt: 疾病名称列表
- circname.txt: circRNA 名称列表