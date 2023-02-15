%使WiFi与zigbee重合子载波数据幅度能量为低
%------------QAM------------------------------
%.设置用的QAM：16,64,256
%---------------------------------------------
%------------area:子载波重合的区域-------------
%1 'area1' %重叠的区间1子载波
%2 'area2'  %重叠的区间2子载波
%3 'area3'   %重叠的区间3子载波
%4 'area4'   %重叠的区间4子载波
%5 'area1_R1'   %重叠的区间1 往右加一个子载波
%6 'area1_R2'  %重叠的区间1 往右加两个子载波
%7 'area2_L1R1'  %重叠的区间2 左右加一个子载波
%8 'area2_L2R2'  %重叠的区间2 左右加两个子载波
%9 'area3_L1R1'  %重叠的区间3 左右加一个子载波
%10 'area3_L2R2'  %重叠的区间3 往右加两个子载波
%11 'area4_L1'   %重叠的区间4子载波 往左加一个子载波
%12 'area4_L2'   %重叠的区间4子载波 往左加两个子载波
%---------------------------------------------
%--------------coding-------------------------
% QAM16:1/2,3/4
% QAM64:1/2,2/3,3/4,5/6
% QAM256:1/2,2/3,3/4,5/6
%---------------------------------------------
QAM = 256;area=7;coding='3/4';
writeLoc = './data/data256.dat';
%1.设置常量
sim_consts = setConsts();
%2.产生原始数据%QAM16:130bits  QAM64:192bits QAM256:250bits
rawdata = randn(1,82*10) > 0;
%3.计算position和对应的wantbits
[position, wantbits] = calPositionAndBits(sim_consts,QAM,area,coding);
%4.插入数据
allData = insertBits(sim_consts,QAM,position,wantbits,rawdata,coding);
%翻转bit以适应Gnuradio并写入文件
gnuradioPattern(allData,writeLoc);




