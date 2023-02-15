import os
import csv
import subprocess
import sys
import shutil

def readFile():
    return csv.reader((open('crates.csv', 'r')))

def main():
    crates_list = readFile()
    count = 0
    for crate in crates_list:
        if crate[0] == 'Crate Id':
            continue
        count += 1
        # git clone <git url> <dir name>_<version>
        cmd = "git clone {} {}_{}".format(crate[-1].strip(), crate[1].strip(), crate[2].strip());
        cwd = sys.path[0] + "/"
        dir_name = "{}_{}".format(crate[1].strip(), crate[2].strip())
        
        if os.path.exists(cwd + dir_name):
            dir_content = os.listdir(cwd + dir_name)
            if not dir_content or dir_content == ['.git']:
                print("{}. {} is empty.".format(count, dir_name))
                # os.rmdir(cwd + dir_name)
                shutil.rmtree(cwd + dir_name)
            else:
                print("{}. {} already download.".format(count, dir_name))
                continue
        print("{}. {}".format(count, ['git', 'clone', crate[-1].strip(), dir_name]))
        p = subprocess.Popen(['git', 'clone', crate[-1].strip(), dir_name])
        try:
            p.wait(1200)
        except subprocess.TimeoutExpired:
            print("{} download timeout".format(dir_name))
            p.kill()

        # os.system(cmd)
        print()


if __name__ == '__main__':
    main()
    