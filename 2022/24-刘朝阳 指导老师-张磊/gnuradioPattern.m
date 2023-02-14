function [] = gnuradioPattern(allData,writeLoc)
%------------------------------------------------------
%翻转比特流以适应GNURadio并写入文件
len = length(allData);
out_bits = zeros(1,len/8);
for i =1:len/8
    sum = 0;
    for j = 1:8
       add = allData(8*(i-1)+j);
       sum = sum + bitshift(add,j-1);
    end
    out_bits(i) = sum;
end
fid=fopen(writeLoc,'w');
fwrite(fid,out_bits,'uint8');
%------------------------------------------------
end

