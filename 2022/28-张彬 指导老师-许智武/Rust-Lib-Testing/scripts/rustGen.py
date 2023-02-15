#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import getopt
import os
import time
import datetime
import re

BIN = "fuzz-target-generator"
IS_WORKSPACE_MEMBER = 0

def get_root_dir():
    tmpPath = os.path.dirname(os.path.abspath(__file__))
    tmpPath = os.path.dirname(os.path.abspath(tmpPath))
    return tmpPath + "/"

def get_package_name():
    package_name_list = []
    package_raw_line = ""
    with open("Cargo.toml", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if re.match('^name = ".*"', line.strip()):
                package_name = re.findall('^name = "(.*?)"', line)
                package_name_list += package_name
            elif re.match('^members = \[.*\]', line.strip()):
                package_name = re.findall('"(.*?)"', line)
                package_name_list += package_name
            elif re.match('^members = .*', line.strip()):
                package_raw_line = "Multiline-member:"
                
            if package_raw_line != "" and "]" not in package_raw_line:
                package_raw_line = package_raw_line + line.strip()
                
    if package_raw_line != "" and package_name_list == []:
        package_name = re.findall('"(.*?)"', package_raw_line)
        package_name_list += package_name
    
    print(package_name_list)
    if len(package_name_list) >= 0:
        IS_WORKSPACE_MEMBER = 1
        print("\nWhich crate do you want to generate fuzz targets for?")
        print("[0] self-defined")
        i = 1
        for each in package_name_list:
            print("["+str(i)+"] " + each)
            i = i + 1
        
        index = input("Please enter a valid number:")
        
        try:
            index = int(index)
        except:
            print("The input is not a number, but a", type(index))
            exit(-1)
        
        if index < 0 or index >= i:
            print("The input is not valid number.")
            exit(-1)
        if type(index) != int:
            print(type(index))
    else:
        IS_WORKSPACE_MEMBER = 0
        index = 1
    
    ret_name_list = []
    if index == 0:
        user_defined = input("Please enter a valid name:")
        ret_name_list.append(user_defined)
    else:
        ret_name_list.append(package_name_list[index-1])
    return ret_name_list
    
def main():

    if len(sys.argv) !=2:
        print('参数个数错误。')
        exit(-1)

    package_name_list = get_package_name()

    if sys.argv[1] == "init":    
        unbuffer_exist=0
        os.system("cargo clean")
        for cmdpath in os.environ['PATH'].split(':'):
            if os.path.isdir(cmdpath) and 'unbuffer' in os.listdir(cmdpath):
                unbuffer_exist=1
        if unbuffer_exist == 0:
            os.system("cargo doc -v 2>&1 | tee tmp.txt")
        else:
            os.system("unbuffer cargo doc -v 2>&1 | tee tmp.txt")

        saveline = []
        with open("tmp.txt", 'r', encoding="utf-8") as f:
            for line in f.readlines():
                if "Running" and "rustdoc --" in line:
                    tmpline = line.strip()
                    tmpline = tmpline[tmpline.find("rustdoc", 1):-1]
                    for each_package in package_name_list:
                        each_package = each_package.replace('-', '_')
                        if "--crate-name " + each_package in tmpline:
                            saveline.append(tmpline)
        os.system("rm -f tmp.txt")

        if saveline != []:
            for eachline in saveline:
                ROOT_DIR = get_root_dir()
                BIN_DIR = ROOT_DIR + "src/rust/build/x86*/stage2/bin/"
                print("\033[31m     Running \033[0m" + "\033[34m" + eachline.replace("rustdoc", BIN_DIR+BIN, 1) + "\033[0m")
                print("--------------------------------")
                os.system(eachline.replace("rustdoc", BIN_DIR+BIN, 1))
        else:
            print("ERROR: saveline is empty")
    
    elif sys.argv[1] == "target":
        for each_package in package_name_list:
            each_package = each_package.replace('-', '_')
            os.system("fuzzer_scripts -f 10 " + each_package)
            os.system("fuzzer_scripts -b " + each_package)

    elif sys.argv[1] == "fuzz":
        for each_package in package_name_list:
            each_package = each_package.replace('-', '_')
            os.system("fuzzer_scripts -fuzz " + each_package)

    elif sys.argv[1] == "replay":
        for each_package in package_name_list:
            each_package = each_package.replace('-', '_')
            os.system("fuzzer_scripts -r " + each_package)

if __name__ == "__main__":
    main()
