#!/bin/bash

# make clean
make
sudo insmod vmac_udp.ko
# dmesg | tail
# sudo rmmod vmac_udp