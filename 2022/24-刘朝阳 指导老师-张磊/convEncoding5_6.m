function [Before_puncture_masks,Before_puncture_bits] = convEncoding5_6(Before_interleave_masks,Before_interleave_bits)
%5/6编码
%----------------------------------------------------------------------
padding3 = zeros(1,length(Before_interleave_bits)/6);
len = floor(length(Before_interleave_masks)/6)*6;
Before_puncture_masks10 = zeros(6,len/6);
Before_puncture_bits10 = zeros(6,len/6);
for i=1:6
    for j = 1:len/6
        Before_puncture_masks10(i,j) = Before_interleave_masks(6*(j - 1) + i);
        Before_puncture_bits10(i,j) = Before_interleave_bits(6*(j - 1) + i);
    end
end
Before_puncture_masks11 = [Before_puncture_masks10(1:3,1:len/6);padding3;padding3;Before_puncture_masks10(4:5,1:len/6);padding3;padding3;Before_puncture_masks10(6,1:len/6)];
Before_puncture_masks = [Before_puncture_masks11(:)'  Before_interleave_masks(len + 1:length(Before_interleave_masks))];  %整合整数部分 + 冗余
Before_puncture_bits11 = [Before_puncture_bits10(1:3,1:len/6);padding3;padding3;Before_puncture_bits10(4:5,1:len/6);padding3;padding3;Before_puncture_bits10(6,1:len/6)];
Before_puncture_bits = [Before_puncture_bits11(:)'  Before_interleave_bits(len + 1:length(Before_interleave_bits))];  %整合整数部分 + 冗余
%----------------------------------------------------------------------
end

