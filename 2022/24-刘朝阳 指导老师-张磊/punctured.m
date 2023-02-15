function [punctured_bits] = punctured(encoded_bits,coding)
%打孔
%1.区分整数 + 冗余的方法二    问题：究竟是以整个包为单位还是symbol？

if(strcmp(coding,'1/2'))
    punctured_bits = encoded_bits;
    
elseif(strcmp(coding,'2/3'))
    len = floor(encoded_bits/4)*4;
    before_puncture = reshape(encoded_bits,4,len/4);
    int_bits = before_puncture(1:3,1:len/6);

    %puncture the remainder bits
    rem_bits = encoded_bits(len + 1:length(encoded_bits)); %余位
    punctured_bits = [int_bits rem_bits];
    punctured_bits = punctured_bits(:)';
    
elseif(strcmp(coding,'3/4'))
    len = floor(encoded_bits/6)*6;
    before_puncture = reshape(encoded_bits,6,len/6);
    int_bits = [before_puncture(1:3,1:len/6);before_puncture(6,1:len/6)];

    %puncture the remainder bits
    rem_bits = encoded_bits(len + 1:length(encoded_bits)); %余位
    if(len(rem_bits) > 0)
        if(len(rem_bits) <= 3)
            rem_bits = rem_bits(1:len(rem_bits));
        else
            rem_bits = rem_bits(1:3);
        end
    end
    punctured_bits = [int_bits rem_bits];
    punctured_bits = punctured_bits(:)';
    
elseif(strcmp(coding,'5/6'))
    len = floor(encoded_bits/10)*10;
    before_puncture = reshape(encoded_bits,10,len/10);
    int_bits = [before_puncture(1:3,1:len/10);before_puncture(6:7,1:len/10);before_puncture(10,1:len/10)];

    %puncture the remainder bits
    rem_bits = encoded_bits(len + 1:length(encoded_bits)); %余位
    if(len(rem_bits) > 0)
        if(len(rem_bits) <= 3)
            rem_bits = rem_bits(1:len(rem_bits));
        elseif(len(rem_bits) > 3 && len(rem_bits) <= 5)
            rem_bits = rem_bits(1:3);
        elseif(len(rem_bits) > 5 && len(rem_bits) <= 7)
            rem_bits = [rem_bits(1:3) rem_bits(6:len(rem_bits))];
        else
            rem_bits = [rem_bits(1:3) rem_bits(6:7)];
        end
    end
    punctured_bits = [int_bits rem_bits];
    punctured_bits = punctured_bits(:)';    
end
    

end

