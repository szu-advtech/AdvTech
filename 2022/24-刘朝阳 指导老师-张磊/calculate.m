function [position,wantbits] = calculate(sim_consts,Before_puncture_masks,Before_puncture_bits,area)
%CALCULATE 此处显示有关此函数的摘要
    %   此处显示详细说明
    m=1;
%     if(sim_consts.QAM == 16)
%         afterConvNums = 4*48;    %QAM16使用1/2编码
%     elseif(sim_consts.QAM == 64)
%         afterConvNums = 6*48 * 3/2 %QAM64使用3/4编码（3/4 / 1/2）
%     elseif(sim_consts.QAM == 256)
%         afterConvNums = 8*48 * 3/2  %QAM256使用3/4编码（3/4 / 1/2）
%     end
    
for k=1:length(Before_puncture_masks)      %找出mask的位置，与他的一半
    if(Before_puncture_masks(k)==1)
        position(1,m)=ceil(k/2);    %/2卷积编码，matlab初始1，所以用ceil
        position(2,m)=k;
        wantbits(1,m) = Before_puncture_bits(1,k);     %想要卷积出来的关键bits
        m=m+1;
    end
end
end

