
import os, sys, re

curr_path = os.path.abspath(__file__)
curr_dirs = os.path.dirname(curr_path)

def get_value_from_pattern (val, line, pat):
    if val != None:
        return val
    mt = re.search (pat, line)
    if mt == None:
        return None
    mt2 = re.search ('[\d|\.]+', mt.group())
    if mt2 == None:
        return None
    return float(mt2.group())

def get_param_from_file (fpath):
    f = open (fpath, 'r')
    lines = f.readlines ()
    f.close ()
    transaction = None
    committed = None
    elapsed = None
    commitrate =None
    commitratio = None
    memory = None
    dcpmm = None
    clwb = None
    sfence = None
    persist = None
    clwbrate = None
    sfencerate = None
    persistrate = None
    for line in lines:
        transaction = get_value_from_pattern (transaction, line, 'transactions: *[\d|\.]+ ')
        committed = get_value_from_pattern (committed, line, 'committed: *[\d|\.]+ ')
        elapsed = get_value_from_pattern (elapsed, line, 'elapsed: *[\d|\.]+ ')
        clwb = get_value_from_pattern (clwb, line, 'aep_clwb_cnt: *[\d|\.]+ ')
        sfence = get_value_from_pattern (sfence, line, 'aep_sfence_cnt: *[\d|\.]+ ')
        persist = get_value_from_pattern (persist, line, 'aep_persist_size: *[\d|\.]+ ')
        memory = get_value_from_pattern (memory, line, 'memory_size: *[\d|\.]+ ')
        dcpmm = get_value_from_pattern (dcpmm, line, 'aep_memory_size: *[\d|\.]+ ')
    if elapsed == None and transaction == None:
        ret = ['']*13
        return ret
    else:
        commitrate = committed/elapsed/1000000
        commitratio= committed/transaction
        clwbrate = clwb/elapsed
        sfencerate = sfence/elapsed
        persistrate= persist/elapsed
        ret = [str(elapsed),str(committed),str(transaction),str(commitrate),str(commitratio),\
               str(clwb),str(sfence),str(persist),\
               str(clwbrate),str(sfencerate),str(persistrate),str(memory),str(dcpmm)]
        return ret
    pass

def collect_result (relative_result_dir, analysis_result_name):
    head = ['方法','行数','请求数','读比率','偏锋率','事务数','线程数',\
            '运行时间',\
            '总提交事务','总运行事务','提交速率(M/s)','提交成功比率(M/s)',\
            'CLWB使用总量(M)','SFENCE使用总量(M)','持久总量(GB)',\
            'CLWB速率(M/s)','SFENCE速率(M/s)','持久速率(GB/s)',\
            '内存使用量(GB)','持久内存使用量(GB)']
            # '数据表使用量(GB)','索引使用量(GB)','版本库使用量(GB)','日志使用量(GB)','检查点使用量(GB)' 
    fs = os.listdir (curr_dirs+'/'+relative_result_dir)
    fs.sort ()
    output_str = ''
    output_str += '\t'.join(head)+'\n'
    for f in fs:
        ps = f.split ('_')
        array = [ps[0]]
        for i in range (1,len(ps)):
            array.append (str(get_value_from_pattern(None, ps[i], '.*')))
        print (array)
        array += get_param_from_file (curr_dirs+'/'+relative_result_dir+'/'+f)
        print (array)
        output_str += '\t'.join (array)+'\n'
    print (output_str)
    fh = open (curr_dirs+'/'+analysis_result_name, 'w')
    fh.write (output_str)
    fh.close ()
    pass

if __name__ == '__main__':
    collect_result ('result/tpcc', 'tpcc.csv')
    pass

