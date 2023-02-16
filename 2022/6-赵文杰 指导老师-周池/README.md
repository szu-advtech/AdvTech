
## 使用方法
### DFC编译器安装
```makefile
cd compiler/ompi
./configure --prefix="安装路径"
make
make install
```
如果configure和gen_version文件没有权限，则运行chmod +x configure/gen_version赋予运行权限

### 利用DFC编译器将DFC代码转换为标准C代码
```
安装路径/bin/ompicc  -k  -v  -s -g    -I"path-to-dfc" -I"path-to-threadpool"  "program-in-dfc"
```

### 编译选项
```
-k 保留中间文件
-v 输出编译信息
-s 生成的代码可以记录调度信息
-g 绘制DAG
```


