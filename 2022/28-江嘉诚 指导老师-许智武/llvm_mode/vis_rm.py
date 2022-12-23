# -*- coding:utf-8 -*-
import os
for file in os.listdir(r'.'):
    if file.endswith(".png"):
        os.system(f"rm {file}")
