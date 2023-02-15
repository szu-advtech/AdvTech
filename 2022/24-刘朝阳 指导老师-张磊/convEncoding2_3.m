function [Before_puncture_masks,Before_puncture_bits] = convEncoding2_3(Before_interleave_masks,Before_interleave_bits)
%2/3编码
%----------------------------------------------------------------------
padding2 = zeros(1,length(Before_interleave_bits)/3);
len = floor(length(Before_interleave_masks)/3)*3;
Before_puncture_masks7 = zeros(3,len/3);
Before_puncture_bits7 = zeros(3,len/3);
w = 0;
for i=1:3
    for j = 1:len/3
        Before_puncture_masks7(i,j) = Before_interleave_masks(3*(j - 1) + i);
        Before_puncture_bits7(i,j) = Before_interleave_bits(3*(j - 1) + i);
    end
end
Before_puncture_masks8 = [Before_puncture_masks7(1:3,1:length(Before_interleave_masks)/3);padding2];
Before_puncture_masks = [Before_puncture_masks8(:)'  Before_interleave_masks(len + 1:length(Before_interleave_masks))];  %整合整数部分 + 冗余
%数据部分，打孔掉的用0代替
Before_puncture_bits8 = [Before_puncture_bits7(1:3,1:length(Before_interleave_bits)/3);padding2];
Before_puncture_bits = [Before_puncture_bits8(:)'  Before_interleave_bits(len + 1:length(Before_interleave_bits))];  %整合整数部分 + 冗余
%----------------------------------------------------------------------
end

