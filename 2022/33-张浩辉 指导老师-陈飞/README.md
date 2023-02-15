# Introduction

该仓库是深圳大学[计算机前沿技术课程](https://csse.szu.edu.cn/courses/AdvTech2022/)的作业提交仓库，目标是复现前沿工作，该仓库复现的是
Zuoru Yang, Jingwei Li, and Patrick P. C. Lee. [Secure and Lightweight Deduplicated Storage via Shielded Deduplication-Before-Encryption](http://www.cse.cuhk.edu.hk/~pclee/www/pubs/atc22.pdf) USENIX ATC 2022. 提供了更加方便的docker容器启动方式。
源代码地址为[DEBE](https://github.com/yzr95924/DEBE)
# Prerequisites

- docker 环境
- sgx driver
# images
## aesm
镜像地址：registry.cn-hangzhou.aliyuncs.com/zhanghaohui/aesm:02
基础镜像为ubuntu18.04
## sgx
镜像地址：registry.cn-hangzhou.aliyuncs.com/zhanghaohui/sgx:01
代码位置在 /root/test/DEBE
基础镜像为ubuntu20.04,sgx-ssl库是通过宿主机编译后docker cp进镜像内的
# start up
```bash
#aesm镜像
docker pull registry.cn-hangzhou.aliyuncs.com/zhanghaohui/aesm:02
#DEBE兼SGX开发环境
docker pull registry.cn-hangzhou.aliyuncs.com/zhanghaohui/sgx:01

docker volume create --driver local --opt type=tmpfs --opt device=tmpfs --opt o=rw aesmd-socket

#启动aesm
docker run --env http_proxy --env https_proxy --device=/dev/isgx -v /dev/log:/dev/log -v aesmd-socket:/var/run/aesmd -it registry.cn-hangzhou.aliyuncs.com/zhanghaohui/aesm:02
#启动应用程序
docker run --env http_proxy --env https_proxy --device=/dev/isgx -v aesmd-socket:/var/run/aesmd  -it registry.cn-hangzhou.aliyuncs.com/zhanghaohui/sgx:01
```
注意 --device是SGX硬件设备的设备文件，不同机器可能会不一样，但是主要是分为两种
一种为`--device=/dev/sgx/enclave --device=/dev/sgx/provision` 配合一起使用
另一种为`--device=/dev/isgx`单独使用

然后就可以在sgx容器中使用DEBE了
