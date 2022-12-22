# MaterialDecomposition

#### 路径指定
在开始一切之前，可以先修改config.json文件中的路径<br>
train_path是训练数据集的路径<br>
test_path是测试数据集的路径<br>
save_path是模型保存的路径<br>
test_img_save_path是利用已有模型在测试集上测试后保存图片的路径<br>

#### 训练
在运行训练代码之前，先使用下面的命令创建一个save_models的文件夹：
```
mkdir save_models
```
然后直接运行main.py文件，就可以开始训练了

#### 测试
运行test_group_data.py文件就可以在测试数据集上进行测试了<br>
需要注意的是，假如你想在数据集中取n张图片，就把utils.py的random_num改成n就行，注意这个n不能超过273<br>
结果会保存在test_img_save_path下<br>

#### Draw the loss curve
This repository did not contain the function of drawing loss because I don't know how to check the tensorboard in my server.
So I write the 'writeTensorboard.py', so that we could create the tensorboardX files under the directory named 'run'.
Then you can run the tensorboard through 'tensorboard --logdir=xxx --host=127.0.0.1'
Note: Before you run the 'writeTensorboard.py', please check you have already remove the blank line in the end of 'loss_train.txt' and 'loss_test.txt'(Both two files would be created during training) 
