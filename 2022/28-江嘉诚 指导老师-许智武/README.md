# CollAFL
本次复现的论文题目为[CollAFL: Path Sensitive Fuzzing](https://ieeexplore.ieee.org/abstract/document/8418631)

本次复现主要修改了AFL源码中afl-fuzz.c文件以及llvm_mode文件夹下的afl-llvm-pass.so.cc文件
### Install

如果使用[docker](https://www.docker.com/), 运行以下命令来构建镜像

```sh
git clone https://github.com/Eterniter/CollAFL-2.57b
cd CollAFL-2.57b
# build docker image
docker build -t collafl --no-cache ./
```
如果不使用docker，手动编译整个项目，一个示例如下所示
```sh
cp /etc/apt/sources.list /etc/apt/sources.list.bak 
sed -i "s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g" /etc/apt/sources.list 
apt-get clean 
apt-get update --fix-missing 
apt-get install -y wget sudo build-essential libacl1-dev git --fix-missing 

apt-get install -y clang-11 llvm-11 --fix-missing 
ln -s /usr/bin/clang-11 /usr/bin/clang 
ln -s /usr/bin/clang++-11 /usr/bin/clang++ 
ln -s /usr/bin/llvm-ar-11 /usr/bin/llvm-ar 
ln -s /usr/bin/llvm-as-11 /usr/bin/llvm-as 
ln -s /usr/bin/llvm-config-11 /usr/bin/llvm-config 

git clone https://github.com/Eterniter/CollAFL-2.57b
cd CollAFL-2.57b
make 
cd llvm_mode 
make 
cd ..
make install 
```


### Example

运行以下命令来测试[LAVA-M](https://ieeexplore.ieee.org/document/7546498)数据集的'base64'程序


```sh
cd CollAFL-2.57b/test
sed -i 's/\r$//' ./fuzz_LAMA-M_base64.sh
sh ./fuzz_LAMA-M_base64.sh
```

