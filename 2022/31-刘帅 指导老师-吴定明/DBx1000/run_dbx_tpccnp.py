
## 此脚本用于批量运行rundb测试程序

import os,re,datetime,sys,time

curr_path = os.path.abspath(__file__)
curr_dirs = os.path.dirname(curr_path)
work_dirs = os.path.join(curr_dirs, '../cicada-engine/build')
dbx_dirs  = curr_dirs
# mode = ['test','CWBL','CZEN']

def cmake_with_param (mode):
    print ('[消息]：cmake ', mode)
    os.chdir (work_dirs)
    cmake_command = 'cmake .. -D'+mode+'=ON'
    os.system (cmake_command)
    if os.path.exists (work_dirs+'/'+'Makefile'):
       return True
    else:
       print ('[错误]：cmake错误，Makefile不存在')
       return False

def remove_cmake ():
    print ('[消息]：remove cmake ')
    os.chdir (work_dirs)
    os.system ('rm -r CMake*')
    os.system ('rm -r cmake*')
    os.system ('rm Makefile')
    if not os.path.exists (work_dirs+'/'+'Makefile'):
       return True
    else:
       print ('[错误]：remove cmake错误，Makefile存在')
       return False

def make_with_no_param ():
    print ('[消息]：make ')
    os.chdir (work_dirs)
    make_command = 'make -j20'
    os.system (make_command)
    if os.path.exists (work_dirs+'/'+'test_tx'):
       return True
    else:
       print ('[错误]：make错误，test_tx不存在')
       return False

def remove_make ():
    print ('[消息]：remove make ')
    os.chdir (work_dirs)
    make_command = 'make clean'
    os.system (make_command)
    if not os.path.exists (work_dirs+'/'+'test_tx'):
       return True
    else:
       print ('[错误]：make clean错误，test_tx存在')
       return False

# '16' '16'  ['mmdb','wbl','zen']  'result'
def run_with_param (wh_num, thread_cnt, prefix_mode, relative_result_dir):
    # 修改config.h运行时参数并编译
    os.chdir (dbx_dirs)
    data = ''
    with open('config.h', 'r') as f:
        for line in f.readlines():
            if (line.startswith('#define THREAD_CNT') == True):
                line = '#define THREAD_CNT ' + str(thread_cnt) + '\n'
            elif (line.startswith('#define PART_CNT') == True):
                line = '#define PART_CNT ' + str(wh_num) + '\n'
            elif (line.startswith('#define VIRTUAL_PART_CNT') == True):
                line = '#define VIRTUAL_PART_CNT ' + str(wh_num) + '\n'
            elif (line.startswith('#define NUM_WH') == True):
                line = '#define NUM_WH ' + str(wh_num) + '\n'
            data += line
    with open('config.h', 'w') as f:
        f.writelines(data)
    os.system ('make -j16')
 
    p1 = 'warehouse@'+str(wh_num)
    p2 = 'threadcnt@'+str(thread_cnt)

    params = [p1, p2]
    result_file_name = prefix_mode+'_'+'_'.join(params)

    result_file_dirs = curr_dirs+'/'+relative_result_dir
    if not os.path.exists (result_file_dirs):
        os.system ('mkdir '+result_file_dirs)

    result_file_path = result_file_dirs+'/'+result_file_name
    if os.path.exists (result_file_path):
        print ('[消息]：文件已存在，跳过运行！', result_file_name)
        return 

    run_command = './rundb > '+result_file_path
    print ('[消息]：运行命令 %s' % run_command)
    try:
        os.system (run_command)
        os.system ('make clean')
    except Exception as e:
        print ('[错误]：rundb程序运行错误！')
        print (e)
    pass

# 定义要运行的测试
def test_frame_1 (prefix_mode_input, relative_result_dir_input):
    prefix_mode = prefix_mode_input
    relative_result_dir = relative_result_dir_input

    if prefix_mode == 'mmdb':
        mode = 'test'
    elif prefix_mode == 'wbl':
        mode = 'CWBL'
    elif prefix_mode == 'zen':
        mode = 'CZEN'
    else:
        print ('[错误]：test_frame输入参数错误！')
        return 

    # 修改Makefile运行时参数并编译
    os.chdir (dbx_dirs)
    data = ''
    with open('Makefile', 'r') as f:
        for line in f.readlines():
            if (line.startswith('CFLAGS +=') == True):
                if prefix_mode == 'mmdb':
                    line = 'CFLAGS += $(INCLUDE) -D test=ON -D NOGRAPHITE=1 -Wno-unused-function -fPIC -no-pie -O3\n'
                elif prefix_mode == 'wbl':
                    line = 'CFLAGS += $(INCLUDE) -D CWBL=ON -D NOGRAPHITE=1 -Wno-unused-function -fPIC -no-pie -O3\n'
                elif prefix_mode == 'zen':
                    line = 'CFLAGS += $(INCLUDE) -D CZEN=ON -D CACHE_QUEUE=ON -D NOGRAPHITE=1 -Wno-unused-function -fPIC -no-pie -O3\n'
                else:
                    print ('[错误]：test_frame输入参数错误！')
                    return 
               
            data += line
    with open('Makefile', 'w') as f:
        f.writelines(data)

    cmake_with_param (mode)
    make_with_no_param ()

    rank = 1
    warehouse_and_thread = [[1,16],[16,16],[128,16]]

    total = len(warehouse_and_thread)
    for wt in warehouse_and_thread:
        wh = wt[0]
        th = wt[1]

        starttime = datetime.datetime.now()
        print ('[%d/%d] 开始时间：%s'%(rank,total,starttime))
        run_with_param (wh, th, prefix_mode,relative_result_dir)
        endtime = datetime.datetime.now()
        usetime = (endtime - starttime).seconds
        print ('[%d/%d] 结束时间时间：%s 花费时间：%d 秒'%(rank,total,endtime,usetime))
        rank += 1

    remove_make ()
    remove_cmake ()
    pass

def test_1 (prefix_mode='mmdb|wbl|zen', relative_result_dir='result/tpcc'):
    run_param = prefix_mode.split('|')
    rank = 1
    for param in run_param:
        print ('[当前组数：%d/实验组数：%d]：%s'%(rank,len(run_param),param))
        test_frame_1 (param, relative_result_dir)
        rank += 1
    pass

if __name__ == '__main__':
    print ('[消息]: 当前路径:', curr_path)
    print ('[消息]: 当前目录:', curr_dirs)
    print ('[消息]: 工作目录1:',work_dirs)
    print ('[消息]: 工作目录2:', dbx_dirs)
    test_1 ()
    pass

