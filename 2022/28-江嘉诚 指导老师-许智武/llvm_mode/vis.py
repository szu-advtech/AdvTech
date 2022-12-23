# -*- coding:utf-8 -*-
import os
file="main.c"
file_name="".join(file.split('.')[:-1])
extension=file.split('.')[-1]
command=f"""
clang `llvm-config --cxxflags` -Wl,-znodelete -fno-rtti -fPIC -shared afl-llvm-pass.so.cc -o afl-llvm-pass.so `llvm-config --ldflags`
clang -Xclang -load -Xclang ./afl-llvm-pass.so -S -emit-llvm {file} -o {file_name}.ll
opt -dot-cfg {file_name}.ll > /dev/null
opt -dot-callgraph {file_name}.ll > /dev/null
"""
os.system(command)
# print(command)
for file in os.listdir(r'.'):
    if file.endswith(".dot"):
        # print(file)
        file_name="".join(file.split('.')[:-1])
        os.system(f"dot -Tpng -o {file_name}.png {file}")
        os.system(f"rm {file}")
os.system("open .")
