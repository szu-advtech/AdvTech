#!/bin/bash
#同理不能直接运行可执行文件
#./home/pi/raspberry_init/modified_V-MAC-Userspace-master/a.out c
RUN_PATH=/home/pi/Desktop/VMAC-Exp/vmac-userspace/
cd $RUN_PATH
gcc stress-test.c csiphash.c vmac-usrsp.c -lpthread -lm
# 给consumer自启动， producer手动
sudo ./a.out c
