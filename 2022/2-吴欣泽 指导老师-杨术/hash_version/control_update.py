#!/usr/bin/env python
# coding=utf-8

import os
import time

# record_period time
record_time = 2

error_hint_str = (
    "Please check whether the switch "
    "is well configured and running."
)


reset_cache_cmd = ( #未能实现功能的命令，只重置了某一个bit
    '''
    echo "register_write ip_to_hc2 22800 0" | \
    /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI \
    hop_count.json 22223
    '''
)

reset_nums_cmd = (
    '''
    echo "register_write reg_update_num 0 0" | \
    /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI \
    hop_count.json 22223
    '''
)

def reset_cache():
    result1 = os.popen(reset_cache_cmd).read() 
    result2 = os.popen(reset_nums_cmd).read()
    if "Done" in result1 and "Done" in result2:
	print "reset successfully! New stage begins!\n\n\n\n"
        return 0
    else:
        print "Error: Can't reset ip_to_hc2 or reg_update_num !\n"
        print error_hint_str
        return -1


def read_num_reg():
    cmd='echo "register_read reg_update_num 0 " | /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI hop_count.json 22223'
    result = os.popen(cmd).read()
    try:
        num_reg_str = result[result.index("reg_update_num[0]="):].split()[1]
        num_reg = int(num_reg_str)
    except:
        print "Error: Cannot read register reg_update_num!\n"
        print error_hint_str
        return -1
    else:
	print "Record nums = "+num_reg_str
        return num_reg

def read_reg_record_ip_to_hc(i):
    cmd='echo "register_read reg_record_ip_to_hc '+ str(i) + '" | /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI hop_count.json 22223'
    result = os.popen(cmd).read()
    try:
        index_str = result[result.index("reg_record_ip_to_hc["+str(i)+"]="):].split()[1]
        index = int(index_str)
    except:
        print "Error: Can't read register reg_record_ip_to_hc!\n"
        print error_hint_str
        return -1
    else:
        return index


#双倍寄存器空间来表示时间维度上的表更新，这里实现不完整，readme中有说明了
def update_register(nums): 
    for i in range(0,nums):
	index = read_reg_record_ip_to_hc(i)
	read_cmd = 'echo "register_read ip_to_hc2 ' + str(index) + '" | /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI hop_count.json 22223'
	result1 = os.popen(read_cmd).read()
	if "Done" in result1:
	    pass
	else:
            print "Error: Can't read register ip_to_hc2 !\n"
            print error_hint_str
            return -1

	write_cmd = 'echo "register_write ip_to_hc ' + str(index) + '" | /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI hop_count.json 22223'
	result2 = os.popen(write_cmd).read()
	if "Done" in result2:
	    pass
	else:
            print "Error: Can't write register ip_to_hc !\n"
            print error_hint_str
            return -1
    print "Debug: update successfully!"
    return 0


#因为输入指令太麻烦了,专门用来测试，查看一些寄存器状态来判断自己的P4代码执行是否正确
def test():  
    cmd='echo "register_read ip_to_hc2 22800'+ '" | /home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI hop_count.json 22223'
    result = os.popen(cmd).read()
    try:
        result_str = result[result.index("ip_to_hc2[22800]="):].split()[1]
        index = int(result_str)
    except:
        print "Error: Can't read register test !\n"
	print result
        print error_hint_str
        return -1
    else:
        return index

def main():
    reset_cache()
    nums=0
    while True:
	print "test1: "+str(test())
	update_register(nums)
        time.sleep(record_time)
	nums=read_num_reg()
	print "test2: "+str(test())
	reset_cache()


if __name__ == "__main__":
    main()


