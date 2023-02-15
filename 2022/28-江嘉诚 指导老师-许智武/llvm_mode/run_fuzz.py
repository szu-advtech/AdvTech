# -*- coding:utf-8 -*-
import os
command=f"""
export LLVM_HOME=/home/hesitater/Study-me/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04/bin
export PATH=$LLVM_HOME:$PATH
../afl-clang-fast main.c -o main
rm -rf fuzz_in
rm -rf fuzz_out
mkdir fuzz_in fuzz_out
echo '1+1' > fuzz_in/seed
echo core >/proc/sys/kernel/core_pattern
../afl-fuzz -i fuzz_in -o fuzz_out ./main
"""
os.system(command)