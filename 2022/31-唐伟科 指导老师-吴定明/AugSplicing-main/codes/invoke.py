import os
import shutil
from codes import util
import codes.CalTopkBlocks as Caltopk


#inputfile是输入张量
#outpath是输出路径
#s: 时间跨度
#k: the number of top blocks
#l: slack constant
#maxSp: the maximum splicing number at each epoch
#N dimension of input data
#steps: the number of time steps
def optimiAlgo(inputfile, outpath, s, k, l, maxSp, N, delimeter, steps):
    augmented_lines, accum_lines = [], []
    sindex = 0
    mints = 0
    with open(inputfile, 'r') as f:
        for line in f:
            #8244,14900,0,1: data of inputfile
            #strip(): remove whitespace around data
            user, obj, ts, v = map(int, line.strip().split(delimeter))
            #将拼接过的行加入augented_lines
            augmented_lines.append(line)
            accum_lines.append(line)
            #splic data according to s
            if ts - mints >= s:
                #os.path.join(): the function is splicing path
                dcube_file = os.path.join('augfile.txt')
                f = open(dcube_file, 'w')
                #将时间在mints～s的数据存入dcube_file
                f.writelines(augmented_lines)
                f.close()
                dcube_output = os.path.join('augfile_dcube_output', str(sindex))
                #os.path.exists(path):判断path路径是否存在
                if not os.path.exists(dcube_output):
                    os.makedirs(dcube_output)
                os.system('cd ./dcube-master && ./run_single.sh' + ' ../' +
                          dcube_file + ' ../' + dcube_output + ' ' + str(N) + ' ari density ' + str(k+l))
                if sindex == 0:
                    curr_output = os.path.join(outpath, str(sindex))
                    if not os.path.exists(curr_output):
                        os.makedirs(curr_output)
                    #得到dcube_output下面所有的文件夹名称
                    for fn in os.listdir(dcube_output):
                        file = os.path.join(dcube_output, fn)
                        file2 = os.path.join(curr_output, fn)
                        #将file中的文件复制到file2中
                        shutil.copy(file, file2)
                else:
                    #取当前时间区间的块
                    dcubeBlocks = util.readBlocksfromPath(dcube_output, k+l)
                    past_output = os.path.join(outpath, str(sindex - 1))
                    #取之前的密集块
                    pastBlocks = util.readBlocksfromPath(past_output, k+l)

                    curr_output = os.path.join(outpath, str(sindex))
                    if not os.path.exists(curr_output):
                        os.mkdir(curr_output)
                    #find top k dense subblocks
                    currBlocks = Caltopk.calTopkBlocks(dcubeBlocks, pastBlocks, k, l, maxSp, N)
                    print("\n\nrunning the AugSplicing algorithm")
                    for idx, block in enumerate(currBlocks):
                        tuplefile = 'block_' + str(idx + 1) + '.tuples'
                        print("block: " + str(idx + 1))
                        #将文件中的每一个块写入指定文件
                        util.writeBlockToFile(curr_output, block, tuplefile)
                mints = ts
                augmented_lines = []
                sindex += 1
                if sindex == steps:
                    print('***end***')
                    break