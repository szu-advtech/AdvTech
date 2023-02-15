#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import getopt
import os
import time
import datetime
import re

def check_rs_file(filename):
    if os.path.splitext(filename)[-1] == ".rs":
        if os.path.exists(filename):
            return True
        else:
            print("\033[31mThe file/folder is not exist.\033[0m")
            return False
    else:
        print("\033[31mThe input file is not a .rs file\033[0m")
        return False

def pollute_single_file(filename):
    print("\033[31m" + filename + ":\033[0m")
    with open(filename, 'r', encoding="utf-8") as f1, open("%s.bak" % filename, "w", encoding="utf-8") as f2:
        ignore = False
        for line in f1:
            space_before = ""
            if re.match("^impl[\s<].*", line.strip()):
                space_before = line[:line.find("impl",1)]
                if "{}" not in line:
                    ignore = True
            elif re.match("^"+space_before+"}$", line):
                ignore = False
            
            if re.match("^crate trait[\s<].*", line.strip()) or re.match("^pub trait[\s<].*", line.strip()):
                space_before_impl = line[:line.find("crate trait",1)]
                ignore = True
            elif re.match("^"+space_before+"}$", line):
                ignore = False

            if re.match("^pub?(\(crate\)\s|\(super\)\s|\s).*", line.strip()):
                if re.match("^pub?(\(crate\)\s|\(super\)\s|\s)fn[\s<].*(.*).*{", line.strip()):
                    if ignore == True:
                        ignore = False
                    f2.write(line)
                else:
                    if "pub(crate)" in line:
                        f2.write(line.replace("pub(crate)", "pub", 1))
                    elif "pub(super)" in line:
                        f2.write(line.replace("pub(super)", "pub", 1))
                    else:
                        f2.write(line)
            elif re.match("^fn[\s<].*(.*).*{", line.strip()):
                if ignore == False:
                    f2.write(line.replace("fn", "pub fn", 1))
                    print("\t\033[31m" + "pub " + line.strip() + "\033[0m")
                else:
                    f2.write(line)
            elif re.match("^crate struct\s.*{", line.strip()) or re.match("^crate struct\s.*;", line.strip()):
                if ignore == False:
                    f2.write(line.replace("crate", "pub", 1))
                    print("\t\033[31m" + line.strip().replace("crate", "pub") + "\033[0m")
                else:
                    f2.write(line)
            elif re.match("^crate fn\s.*{", line.strip()) or re.match("^crate fn\s.*;", line.strip()) or re.match("^crate fn\s.*\(", line.strip()):
                f2.write(line.replace("crate", "pub", 1))
                print("\t\033[31m" + line.strip().replace("crate", "pub") + "\033[0m")
            elif re.match("^crate enum\s.*{", line.strip()):
                if ignore == False:
                    f2.write(line.replace("crate", "pub", 1))
                    print("\t\033[31m" + line.strip().replace("crate", "pub") + "\033[0m")
                else:
                    f2.write(line)
            elif re.match("^struct\s.*{", line.strip()) or re.match("^struct\s.*;", line.strip()):
                if ignore == False:
                    f2.write(line.replace("struct", "pub struct", 1))
                    print("\t\033[31m" + line.strip().replace("struct", "pub struct") + "\033[0m")
                else:
                    f2.write(line)
            elif re.match("^enum\s.*{", line.strip()):
                f2.write(line.replace("enum", "pub enum", 1))
                print("\t\033[31m" + line.strip().replace("enum", "pub enum") + "\033[0m")
            # crate name : struct, crate fn..
            elif re.match("^crate\s.*{", line.strip()) or re.match("^crate\s.*,", line.strip()) or re.match("^crate\s.*;", line.strip()):
                f2.write(line.replace("crate", "pub", 1))
                print("\t\033[31m" + line.strip().replace("crate", "pub") + "\033[0m")
            elif re.match("^pub(crate)\s.*{", line.strip()) or re.match("^pub(crate)\s.*,", line.strip()) or re.match("^pub(crate)\s.*;", line.strip()):
                f2.write(line.replace("pub(crate)", "pub", 1))
                print("\t\033[31m" + line.strip().replace("pub(crate)", "pub") + "\033[0m")
            elif re.match("^mod\s.*;", line.strip()):
                    f2.write(line.replace("mod", "pub mod", 1))
                    print("\t\033[31m" + line.strip().replace("mod", "pub mod") + "\033[0m")
            else:
                f2.write(line)

                    
    os.remove(filename)
    os.rename("%s.bak" % filename, filename)
    print("")

def pollute_folder(foldername):
    for root,dirs,files in os.walk(foldername):
        for file in files:
            if os.path.splitext(file)[-1] == ".rs":
                filename = os.path.join(root,file)
                pollute_single_file(filename)
        
def main(argv):
    #通过 getopt模块 来识别参数demo
    TargetFile = ""
    Folder = ""

    try:
        opts, args = getopt.getopt(argv, "hf:t:", ["help", "Folder", "TargetFile="])

    except getopt.GetoptError:
        print('Error: ExtractLine.py -f <TargetFile>')
        sys.exit(2)

    # 处理 返回值options是以元组为元素的列表。
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("./rs_polution.py -f <Folder> or ./rs_polution.py -t <TargetFile>")
            sys.exit()
        elif opt in ("-f", "--folder"):
            Folder = arg
        elif opt in ("-t", "--file"):
            TargetFile = arg
        
    if TargetFile == "" and Folder == "":
        print("\033[31mError: Command line is empty.\033[0m")
        print("\033[31mTips: Using -h to view help.\033[0m")
        sys.exit(2)
    elif TargetFile != "" and Folder != "":
        print("\033[31mError: Do not use '-f' and '-t' at the same time.\033[0m")
        print("\033[31mTips: Using -h to view help.\033[0m")
        sys.exit(2)
    elif TargetFile != "" and Folder == "":
        print("Target file = ", TargetFile)
    else:
        print("Folder = ", Folder)
    print('')        

    # 打印 返回值args列表，即其中的元素是那些不含'-'或'--'的参数。
    for i in range(0, len(args)):
        print('参数 %s 为：%s' % (i + 1, args[i]))

    # start
    if TargetFile != "" and Folder == "":
        if check_rs_file(TargetFile):
            pollute_single_file(TargetFile)
    else:
        pollute_folder(Folder)
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
