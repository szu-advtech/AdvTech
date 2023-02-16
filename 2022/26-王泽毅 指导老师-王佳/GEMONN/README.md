# GEMONN

## Requirements and Dependency
```bash
torch>=1.4.0
torchvision >=0.5.1
cuda 10.0
cupy
numpy
matplotlib
```

## python GEMONN/Main.py --Generations 500 --PopulationSize 50 --HiddenNum 500 --plot --save --save_dir ./result
## python GEMONN/Contrast_Algorithm.py --HiddenNum 500 --plot --save --save_dir ./result

## Result 
# plot_xxx.png是在MNIST结果上带有sinlge隐藏层的自动编码器（AE）的结果：x轴是MSE损耗，y轴是L0范数的稀疏性
# MSE_Pareto显示了GEMONN，SGD，Adam和NSGA-II在MNIST上获得的最终解决方案的比较，显示在MSE-稀疏性轴上
# MSE_descent.png比较了GEMONN的收敛性，并比较了提出这些MSE下降的方法（MNIST）
# sparsity_comparison.png显示了最终解决方案在稀疏性方面的比较
# xxxst_weights.png显示了GEMONN在不同世代（第 1、10、100 和 500 代）中获得的稀疏性

## Extension
# 训练模型，如LSTM和CNN，在Private_function.py文件
# 1.从pytorch中得到模型 Model = LeNet()
# 2.获取模型的权重 Parameter_dict = Model.state_dict()
# 3.初始化总体并获得模型不同部分的相应大小和长度信息 Population, Boundary, Coding, SizeInform, LengthInform = Initialization_Pop_(PopSize =10, Model = Model)
# 4.获取总体中每个个体的权重字典并计算损失以进行评估 Parameter_dict_i = Pop2weights(Population[0], SizeInform, LengthInform, Parameter_dict)
# 5.由稀疏SGD或spare-Adam支持的GEMONN训练此模型
