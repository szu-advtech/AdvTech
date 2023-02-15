#!/bin/bash

# 实际工作中的 shell 脚本，其所在的目录中可能包含该脚本执行所需要的文件和工具。
# 除非在 shell 脚本所在目录中运行脚本，否则 shell 脚本将找不到它所依赖的文件和工具。

tmp_path=`pwd`
# ko position
CUR_PATH=/home/pi/Desktop/VMAC-New-Kernel/pi4
cd $CUR_PATH
sudo rmmod ath9k_htc
sudo rmmod ath9k_common
sudo rmmod ath9k_hw
sudo rmmod ath
sudo rmmod mac80211
sudo insmod ath.ko
sudo insmod vmac.ko
sudo insmod ath9k_hw.ko
sudo insmod ath9k_common.ko
sudo insmod ath9k_htc.ko
cd $tmp_path
# userspace position
RUN_PATH=/home/pi/Desktop/VMAC-Exp/vmac-userspace
cd $RUN_PATH
gcc stress-test.c csiphash.c vmac-usrsp.c -lpthread -lm
# 给consumer自启动， producer手动
sudo ./a.out c
