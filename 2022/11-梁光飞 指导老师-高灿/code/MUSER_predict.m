%{
This is the predict code for MUSER algorithm.

Input:
    data(MLAI structure):
        data:arranged by column

Output:
    pre: prediction

Related paper:
Partial Multi-Label Learning via Multi-Subspace Representation

Author: coffee
Date: 2022.11.30
%}

function [pre] = MUSER_predict(data,pars)
    if data.infm.sample_order == 'row'
        X = data.data';
    else
        X = data.data;
    end
    pre = X'*pars.Q*pars.W*pars.P;
end