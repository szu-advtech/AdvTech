function [position, wantbits] = calPositionAndBits(sim_consts,QAM,area,coding)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    sim_consts.QAM = QAM;
    sim_consts.block_size = log2(QAM)*48;
    matrix_deinterleave = deinterleave(sim_consts,QAM);
    subcarriers = calSubcarriers(sim_consts,area);
    if(sim_consts.QAM == 16)
        %计算出交织前的比特以及重要比特位
       [Before_interleave_masks,Before_interleave_bits] = beforeInterleaveQAM16(sim_consts,matrix_deinterleave,subcarriers);
       %经过验证，QAM16的1/2编码与3/4编码的position位效率一致，都是2个重复，所以采用1/2编码,所以不处理

    elseif(sim_consts.QAM == 64)
       %计算出交织前的比特以及重要比特位
       [Before_interleave_masks,Before_interleave_bits] = beforeInterleaveQAM64(sim_consts,matrix_deinterleave,subcarriers);

    elseif(sim_consts.QAM == 256)
       %计算出交织前的比特以及重要比特位
       [Before_interleave_masks,Before_interleave_bits] = beforeInterleaveQAM256(sim_consts,matrix_deinterleave,subcarriers);
    end
    if(strcmp(coding,'1/2'))
        Before_puncture_masks = Before_interleave_masks;
        Before_puncture_bits = Before_interleave_bits;
    elseif(strcmp(coding,'2/3'))
        [Before_puncture_masks,Before_puncture_bits] = convEncoding2_3(Before_interleave_masks,Before_interleave_bits);
    elseif(strcmp(coding,'3/4'))
        [Before_puncture_masks,Before_puncture_bits] = convEncoding3_4(Before_interleave_masks,Before_interleave_bits);
    elseif(strcmp(coding,'5/6'))
        [Before_puncture_masks,Before_puncture_bits] = convEncoding5_6(Before_interleave_masks,Before_interleave_bits);
    end
    
   %计算出position与wantbits
   [position,wantbits] = calculate(sim_consts,Before_puncture_masks,Before_puncture_bits,area);
end

