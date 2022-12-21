### 1. 环境：安装Python, 安装相关依赖包
    #Python: 我的版本是3.9.13(不一样应该也可以)
    pip install -r requirements.txt
### 2. 运行：在Windows命令窗口cmd下运行下面命令
    python main.py
### 3. 结果
    程序输出结果放在output/images中，以程序运行时间(如2022-12-18 21_47_53)作为文件夹名称。
    各种算法(Entropic Peeling, Entropic Enumeration...)的运行结果均以折线图的方式存放在目录中。
### 4. 参考文献
#### 4.1 Entropic Peeling, Entropic Enumeration Algorithm参考文献：
        论文名：Entropic Causal Inference: Graph Identifiability
        URL：https://proceedings.mlr.press/v162/compton22a.html
#### 4.2 Joint Entropy Minimization Algorithm参考文献：
        论文名：Entropic Causal Inference
        URL：https://arxiv.org/abs/1611.04035
### 5. 说明
    1. Entropic Enumeration的实现我写的一般，只能跑小规模的图(18条边以下)，跑不了中等规模的图(medium graphs)，因为耗时会很长，机器配置不好容易内存溢出。
    2. 本文算法的实现均是基于我个人对原论文的理解，因为本人能力和水平十分有限，如果你觉得我写错了，那么你是对的，还请各位多多包涵。