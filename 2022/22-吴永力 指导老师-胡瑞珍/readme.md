
## Run 

```
pip install -r requirements.txt
```

按照目前的复现的，目前提供了两个场景：

1. 三摆场景：

   ```
   python demo_threePendulum.py
   ```

   ![image-20221207195244207](https://img2023.cnblogs.com/blog/1656870/202212/1656870-20221207195244537-1257811328.png)

2. 盒子：

   ```
   python demo_box.py
   ```

   ![image-20221207195430151](https://img2023.cnblogs.com/blog/1656870/202212/1656870-20221207195430462-1631538396.png)

由于笔者采用GS迭代，但是又没有实现并行化版本的GS,在多个约束或者多次迭代时候很多概率约束求解不正确，目前场景提供的标准的numsubStep=5时是效果最好的。（CPU并行）

### Benchmark(估摸)

CPU：i7-12700H

GPU：NVIDIA GeForce RTX 3070 Laptop GPU

|                     | numSubStep=5 | numSubStep=10 |
| ------------------- | ------------ | ------------- |
| Boxes(CPU)          | 80FPS        | 60FPS         |
| Boxes(GPU)          | 43FPS        | 30FPS         |
| Three pendulum(CPU) | 60FPS        | 48FPS         |
| Three pendulum(GPU) | 35FPS        | 24FPS         |
