function [matrix_deinterleave] = deinterleave(sim_consts,QAM)
%j_k因为是固定的，所以直接写在了sim_consts里
    % in_bits = randn(1,block_size*1) > 0;
    % % 第一步转置后每行的索引，第一层交织
    % m_k = block_size/N_col*mod([0:block_size-1],N_col)+floor([0:block_size-1]/N_col); %转置为16*18,以列为单位
    % % 第二步转置后每行的索引，第二层交织
    % s = max(N_BPSC/2,1);
    % j_k = s*floor(m_k/s)+mod((m_k+block_size-floor(N_col*m_k/block_size)),s);   %第二层交织：以s*N_col块为单位，每一列进行一定规律的交换

    % 完成交织
    if(QAM == 16)
        j_k = sim_consts.j_kQAM16;
    elseif(QAM == 64)
        j_k = sim_consts.j_kQAM64;
    elseif(QAM == 256)
        j_k = sim_consts.j_kQAM256;
    end
    matrix_deinterleave = zeros(1,sim_consts.block_size);
    for k=1:sim_consts.block_size
        matrix_deinterleave(k)=find(j_k == k-1);
    end
end

