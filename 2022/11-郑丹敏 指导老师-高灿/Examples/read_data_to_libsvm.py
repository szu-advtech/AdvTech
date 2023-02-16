##
# create by zhenyuxiaoge on 2016/11/25
##

# transform data.txt to libsvm type data, get libsvm.txt

# read data file
# 可以将数据转成libsvm-openset-master可以读取的格式
readin = open('../NNDR_feature_vectors/15scenes.dat', 'r')
# write data file
output = open('../NNDR_feature_vectors/libsvm', 'w')
try:
    the_line = readin.readline()
    while the_line:
        # delete the \n
        the_line = the_line.strip('\n')
        index = 0;
        output_line = ''
        for sub_line in the_line.split('\t'):
            # the label col
            if index == 0:
                output_line = sub_line
            # the features cols
            if sub_line != 'NULL' and index != 0:
                the_text = ' ' + str(index) + ':' + sub_line
                output_line = output_line + the_text
            index = index + 1
        output_line = output_line + '\n'
        output.write(output_line)
        the_line = readin.readline()
finally:
    readin.close()

