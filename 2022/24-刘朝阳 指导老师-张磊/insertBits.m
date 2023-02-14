function [insertData] = insertBits(sim_consts,QAM,position,wantbits,rawdata,coding)
%计算需要插入的比特数
%insertLocs = calInsertLocs(position);
insertNums = length(position(1,:));
if(QAM == 16)
    unit = sim_consts.unitBitsBeforeConvQAM16;
    j_k = sim_consts.j_kQAM16;
    overlapBitSame;
elseif(QAM == 64)
    unit = sim_consts.unitBitsBeforeConvQAM64;
    j_k = sim_consts.j_kQAM64;
elseif(QAM == 256)
    unit = sim_consts.unitBitsBeforeConvQAM256;
    j_k = sim_consts.j_kQAM256;
end
unitRawNums = unit - insertNums;
n = ceil(length(rawdata)/unitRawNums);  %分作多少个symbol
overlapBitSameFlag = 1;
symbolBits = sim_consts.num_ofdm * log2(QAM);
%-----------先处理整数---------------
for num = 1:n
    unitdata = rawdata(1+unitRawNums*(num-1):unitRawNums*num);
    for m=1:insertNums
        if(position(1,m)<7)
            tempzero=zeros(1,7-position(1,m));
            tempdata=unitdata((position(1,m)-1):-1:1);
            tempbits=[tempdata tempzero];
        else
            tempbits=unitdata((position(1,m)-1):-1:(position(1,m)-6));
        end
        if(position(2,m) == 2*position(1,m)-1)%significance bit的位置为奇数，也就是用第一个卷积码卷积出来的
             if(m < insertNums && position(1,m) == position(1,m+1))%重合情况下插入n-1位
                if(tempbits(5) == 0 && overlapBitSameFlag == 1)%n-5=0且significant bits相同
                    x = 0;
                elseif(tempbits(5) == 0 && overlapBitSameFlag == 0)%n-5=0且significant bits相反
                    x = 1;
                elseif(tempbits(5) == 1 && overlapBitSameFlag == 1)%n-5=1且significant bits相同
                    x = 1;
                else
                    x = 0;
                end
            unitdata = [unitdata(1:position(1,m) - 2) x unitdata(position(1,m) - 1:length(unitdata))];
            tempbits=unitdata((position(1,m)-1):-1:(position(1,m)-6));  %更新tempbits
             end
            temp1=[0 tempbits];     %前面加个0(因为是翻转的，相当于significance bit位)
            coded_bit1 = rem(sum(temp1.*sim_consts.ConvCodeGenPoly(1,:)),2);   %significance bit为0卷积出来的比特
            if(coded_bit1 == wantbits(m))     %卷积出来的比特值 = 交织前的比特  所以是加0能够能够达到要求
                unitdata = [unitdata(1:position(1,m)-1) 0 unitdata(position(1,m):length(unitdata))];
            else        %用0不能，那就是1
                unitdata = [unitdata(1:position(1,m)-1) 1 unitdata(position(1,m):length(unitdata))];
            end
        else%significance bit的位置为偶数，也就是用第二个卷积码卷积出来的
            %若为第一个significance bit
            if(m == 1)      %为什么m=1是特殊的？跟else的内容不是完全一样？
                temp1=[0 tempbits];     %前面加个0(因为是翻转的，相当于significance bit位)
                coded_bit1 = rem(sum(temp1.*sim_consts.ConvCodeGenPoly(2,:)),2);   %significance bit为0卷积出来的比特
                if(coded_bit1 == wantbits(m))     %卷积出来的比特值 = 交织前的比特  所以是加0能够能够达到要求
                    unitdata = [unitdata(1:position(1,m)-1) 0 unitdata(position(1,m):length(unitdata))];
                else        %用0不能，那就是1
                    unitdata = [unitdata(1:position(1,m)-1) 1 unitdata(position(1,m):length(unitdata))];
                end
            elseif(position(1,m) == position(1,m-1))
                %什么都不做
            else
                temp1=[0 tempbits];     %前面加个0(因为是翻转的，相当于significance bit位)
                coded_bit1 = rem(sum(temp1.*sim_consts.ConvCodeGenPoly(2,:)),2);   %significance bit为0卷积出来的比特
                if(coded_bit1 == wantbits(m))     %卷积出来的比特值 = 交织前的比特  所以是加0能够能够达到要求
                    unitdata = [unitdata(1:position(1,m)-1) 0 unitdata(position(1,m):length(unitdata))];
                else        %用0不能，那就是1
                    unitdata = [unitdata(1:position(1,m)-1) 1 unitdata(position(1,m):length(unitdata))];
                end   
            end
        end 
    end

    %四、将WiFidata送入encoding和interleaving
    encoded_bits1 = DataConvEncoder(unitdata);  %卷积编码
    encoded_bits = encoded_bits1(1:symbolBits*eval(coding)*2);

    %打孔
    punctured_bits = punctured(encoded_bits,coding);
    
    %交织
    interleaved_encoded_bits = punctured_bits(:,j_k+1);
    %interleaved_encoded_bits111=reshape(interleaved_encoded_bits,8,block_size/8);
    %[interleaved_encoded_bits111(2:4,16:21);interleaved_encoded_bits111(6:8,16:21)]
    %interleaved_encoded_bits111(1:8,2:6)
    rawdata(1+symbolBits*(num-1) : symbolBits*num) = interleaved_encoded_bits;
end
insertData = rawdata;
%------------------------------------------------------
end

