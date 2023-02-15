#!/bin/bash

make clean
sudo rmmod vmac_udp
make
sudo insmod vmac_udp.ko
# dmesg | tail