#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import getopt
import os
import time
import datetime
import re

def get_root_dir():
    tmpPath = os.path.dirname(os.path.abspath(__file__))
    tmpPath = os.path.dirname(os.path.abspath(tmpPath))
    return tmpPath + "/"

def copy_fuzz_target_generator_into_tools(ROOT_DIR):
    os.chdir(ROOT_DIR+"src/fuzz-target-generator")
    os.system("cargo clean")
    print("cp -rf " + ROOT_DIR + "src/fuzz-target-generator " + ROOT_DIR + "src/rust/src/tools")
    os.system("cp -rf " + ROOT_DIR + "src/fuzz-target-generator " + ROOT_DIR + "src/rust/src/tools")
    os.chdir(ROOT_DIR+"src/rust/src/tools/fuzz-target-generator")
    os.system("rm -f rust-toolchain && rm -f Cargo.lock && rm -f LICENSE && rm -f README.md && rm -f build.rs")
    filename = "Cargo.toml"
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        for line in f1:
            if 'rustdoc = {path = "../rust/src/librustdoc"}' in line:
                f2.write(line.replace("../rust/src/librustdoc","../../librustdoc", 1))
            else:
                f2.write(line)

    os.remove(filename)
    os.rename("%s.bak" % filename, filename)

def change_the_cargo_toml(ROOT_DIR):
    filename = "Cargo.toml"
    os.chdir(ROOT_DIR+"src/rust")
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        flag = False
        for line in f1:
            if flag != True:
                if "src/tools/fuzz-target-generator" in line:
                    flag = True
                elif "src/tools/rustdoc" in line:
                    f2.write('  "src/tools/fuzz-target-generator",\n')
                    print('  "src/tools/fuzz-target-generator",')
                    flag = True
            f2.write(line)

    os.remove(filename)
    os.rename("%s.bak" % filename, filename)

def remove_extern_in_librs(ROOT_DIR):
    os.chdir(ROOT_DIR+"src/rust/src/tools/fuzz-target-generator/src/bin")
    filename = "fuzz-target-generator.rs"
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        for line in f1:
            if "extern crate rustdoc" in line:
                f2.write(line.replace("extern crate", "use", 1))
                print(line.replace("extern crate", "use", 1), end="")
            else:
                f2.write(line)
    
    os.remove(filename)
    os.rename("%s.bak" % filename, filename)

    filename = "cargo-rsdoc.rs"
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        for line in f1:
            if "extern crate rustdoc" in line:
                f2.write(line.replace("extern crate", "use", 1))
                print(line.replace("extern crate", "use", 1), end="")
            else:
                f2.write(line)
    
    os.remove(filename)
    os.rename("%s.bak" % filename, filename)

def modify_bootstrap_builder(ROOT_DIR):
    os.chdir(ROOT_DIR+"src/rust/src/bootstrap")
    filename = "builder.rs"
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        flag = False
        for line in f1:
            if flag != True:
                if "tool::FuzzTargetGenerator," in line:
                    flag = True
                elif "tool::Rustdoc," in line:
                    f2.write('                tool::FuzzTargetGenerator,\n')
                    print('                tool::FuzzTargetGenerator,')
                    flag = True
            f2.write(line)

    os.remove(filename)
    os.rename("%s.bak" % filename, filename)

def add_fuzz_target_generator_in_bootstrap_tool(curline):
    saveline = []
    for line in curline:
        if "pub struct Rustdoc {" in line:
            saveline.append(line.replace("Rustdoc","FuzzTargetGenerator",1))
        elif "impl Step for Rustdoc {" in line:
            saveline.append(line.replace("Rustdoc","FuzzTargetGenerator",1))
        elif "Rustdoc" in line:
            saveline.append(line.replace("Rustdoc","FuzzTargetGenerator",1))
        elif "src/tools/rustdoc" in line:
            saveline.append(line.replace("src/tools/rustdoc","src/tools/fuzz-target-generator",1))
        elif '.join(exe("rustdoc"' in line:
            saveline.append(line.replace('exe("rustdoc"','exe("fuzz-target-generator"' ,1))
        elif '.join(exe("rustdoc_tool_binary"' in line:
            saveline.append(line.replace('exe("rustdoc_tool_binary"','exe("fuzz-target-generator"' ,1))
        else:
            saveline.append(line)
    return saveline

def modify_bootstrap_tool(ROOT_DIR):
    os.chdir(ROOT_DIR+"src/rust/src/bootstrap")
    filename = "tool.rs"
    saveline = []
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        done = False
        flag = False
        for line in f1:
            if "pub struct FuzzTargetGenerator {" in line:
                done = True
            elif "pub struct Rustdoc {" in line and done == False:
                flag = True
                saveline.append("#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]\n")
            elif "#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]" in line and flag==True:
                flag = False
                if done == False:
                    saveline = add_fuzz_target_generator_in_bootstrap_tool(saveline)
                    for eachline in saveline:
                        f2.write(eachline)
                        print(eachline, end="")
                    done = True

            if flag == True:
                saveline.append(line)
                f2.write(line)
            else:
                f2.write(line)

    os.remove(filename)
    os.rename("%s.bak" % filename, filename)

def main(argv):
    #通过 getopt模块 来识别参数demo
    Folder = ""

    try:
        opts, args = getopt.getopt(argv, "h", ["help"])

    except getopt.GetoptError:
        print('Error: autoConfig.py')
        sys.exit(2)

    # 处理 返回值options是以元组为元素的列表。
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("./autoConfig.py")
            sys.exit()
    print('')        

    # start
    ROOT_DIR = get_root_dir()
    change_the_cargo_toml(ROOT_DIR)
    copy_fuzz_target_generator_into_tools(ROOT_DIR)
    remove_extern_in_librs(ROOT_DIR)
    modify_bootstrap_builder(ROOT_DIR)
    modify_bootstrap_tool(ROOT_DIR)
    # end
	
if __name__ == "__main__":
    # sys.argv[1:]为要处理的参数列表，sys.argv[0]为脚本名，所以用sys.argv[1:]过滤掉脚本名。
    starttime = datetime.datetime.now()
    main(sys.argv[1:])
    endtime = datetime.datetime.now()
    print()
    print("start time: ", starttime)
    print("end time: ", endtime)
    print("@@@ Finished @@@")
