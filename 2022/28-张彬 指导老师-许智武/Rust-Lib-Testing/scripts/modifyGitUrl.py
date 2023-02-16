# 用于修改项目中.gitmodules的URL
# 解决git submodule update十分慢的问题
import os

ROOT_DIR = os.path.abspath(os.curdir)

def main():
    print(ROOT_DIR)
    gitmodules = ROOT_DIR + '/.gitmodules'
    print(gitmodules)
    file_data = ""
    with open(gitmodules, "r", encoding="utf-8") as f:
        for line in f:
            if 'https://github.com' in line:
                print('origin:'+line)
                line = line.replace('https','https://github.91chi.fun/https')
                print('updated:'+line)
            file_data += line
    with open(gitmodules, "w", encoding="utf-8") as f:
        f.write(file_data)
        print("modify .gitmodule success")
    # pass

if __name__ == '__main__':
    main()
    
