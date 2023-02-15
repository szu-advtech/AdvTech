# TPDS-INSPEC-19866720
This code accompanies the paper  "Cost-Effective App Data Distribution in Edge Computing,"[1]  by X. Xia,etc.,INSPEC number is 19866720. This code writed by qiusiki@foxmail.com make for  Advanced Computer Technology course.
In this repository, the features below are implemented:
- reproduct EDDIP、EDDA alogrithms proposed by the paper.
- reproduct referenced constrastive algrithms Random,Greedy, MMR[2]  
- train for EUA dataset : https://github.com/swinedge/eua-dataset

## Usage
for implement this code,you should install at first:
>
> - python >=3.7
> - python library networkx、 matplotlib、 queue
>

I test the reproduction code in pycharm2020.

for single example experiment, you should fir generate a GP model graph by python NetworkX.

###
```html
> run GreatGrapgGM.py 

then input your number of vertex and edges in graph ,then the result will be restored in randomNetwork01.txt.

> or run GreatGrapgGP.py  

then input your number of vertex and densitity in graph ,then the result will be restored in randomNetwork02.txt.

then calculate the cost and  time for EDDIP\EDDA\Random\Greedy\MMR

>  run main.py 

if your want to adjust some parameter of the problems, you can amend this command：

```

```python
g=createGM(yg=20,dlimit=2,rnump=1,readfile='randomNetwork02.txt')
```
### Result of a example 
```html
> run GreateGraphGP.py

> input : 10,1.2 
> output: 

> randomNetwork02.txt
0 1 1 1 0 0 0 1 1 1 
1 0 1 1 0 0 0 0 0 0 
1 1 0 1 0 0 0 0 0 0 
1 1 1 0 1 0 0 0 0 0 
0 0 0 1 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 
2 1 4 8 3 9 5 10 

```
![](https://github.com/Siki-cloud/TPDS-INSPEC-19866720/blob/e48ce4fa0a8541cb54c1b05abb86c11f663834ce/photos/2.png)
```python
>i run main.py<br>
```
```python
>g=createGM(yg=20,dlimit=2,rnump=1,readfile='randomNetwork02.txt')<br>
```
```html
output:
> 物种算法得到的cost 和 计算时间分别为：
> Greedy: cost = 46 time = 0.0
> Random: cost = 225 time = 0.0
> EDDA: cost = 46 time = 0.0019714832305908203
> EDDIP: cost = 46 time = 0.010987043380737305
> MMR: cost = 46 time = 0.005053520202636719
<br>
>  red represences the specified edge server -> r node.
>  green represences the common edge server -> common node.
>  the graph visualization like the below image shows:

```
![](https://github.com/Siki-cloud/TPDS-INSPEC-19866720/blob/e48ce4fa0a8541cb54c1b05abb86c11f663834ce/photos/23.png)

同时，代码的演示视频在images目录下可查看。
## dataset

https://github.com/swinedge/eua-dataset<br>

## Reference
[1]X. Xia, F. Chen, Q. He, J. C. Grundy, M. Abdelrazek and H. Jin, "Cost-Effective App Data Distribution in Edge Computing," in IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 1, pp. 31-44, 1 Jan. 2021, doi: 10.1109/TPDS.2020.3010521.<br>
[2] G. Xue, “Minimum-cost QoS multicast and unicast routing in communication networks,” IEEE Trans. Commun., vol. 51, no. 5, pp. 817–824, May 2003.<br>
