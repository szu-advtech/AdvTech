#### 本文件夹记录了以内核模块运行的类UDP层，编译命令：
```shell
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
```

#### 注意事项
1. 树莓派中必须先存在`raspi-kernel-header`
2. 插入模块和清除模块可查看`debug.sh`和`clear.sh`
3. 该模块运行依赖于`vmac module`
