function [Exactly,condition,citr,slst,X, f, v, Pbestx, Pbestf,b,state] = Global_search(Data,bu,bd,condition,slst,n,termination,citr,X, f, v, Pbestx, Pbestf, Gbestx, Gbestf)
c = size(Data,2)-1;
kernal='cubic';

Bd = repmat(bd,c,1); Bu = repmat(bu,c,1);
%     L = Data(:,1:c); LY = Data(:,c+1);
%     global coefC
%     coefC = rbfcreate(L',LY','RBFFunction', kernal);
%     nameR = @YCRBF;
    
    %% 训练三个代理
    x = Data(:,1:c); y = Data(:,c+1);
    % kriging
    srgtOPTKRG  = srgtsKRGSetOptions(x, y);
    srgtSRGTKRG = srgtsKRGFit(srgtOPTKRG);
    [PRESSRMS_KRG, eXV_KRG] = srgtsCrossValidation(srgtOPTKRG);
    
    % polynomial response surface(PRS)
    srgtOPTPRS  = srgtsPRSSetOptions(x, y);
    srgtSRGTPRS = srgtsPRSFit(srgtOPTPRS);
    [PRESSRMS_PRS, eXV_PRS] = srgtsCrossValidation(srgtOPTPRS);
    
    % radial basis function(RBF)
    srgtOPTRBF  = srgtsRBFSetOptions(x, y);
    srgtSRGTRBF = srgtsRBFFit(srgtOPTRBF);
    [PRESSRMS_RBF, eXV_RBF] = srgtsCrossValidation(srgtOPTRBF);
    
    %% Computing Weights
    eXVMatrix = [eXV_KRG eXV_RBF eXV_PRS];
    CMatrix   = srgtsWASComputeCMatrix(x, eXVMatrix);
    
    srgtsOPTs   = {srgtOPTKRG  srgtOPTRBF  srgtOPTPRS};
    srgtsSRGTs  = {srgtSRGTKRG srgtSRGTRBF srgtSRGTPRS};
    WAS_Model   = 'OWSdiag';
    WAS_Options = CMatrix;
    
    srgtOPTWAS  = srgtsWASSetOptions(srgtsOPTs, srgtsSRGTs, WAS_Model, WAS_Options);
    srgtSRGTWAS = srgtsWASFit(srgtOPTWAS);
    w=srgtSRGTWAS.WAS_Weights;
    
evals = size(Data,1); bound = [bd, bu]; state = zeros(size(slst,2),1);
for i = 1 : size(slst,2)
    if condition(i) == 0 
        [f, X, Pbestx, Pbestf, Gbestx, Gbestf, v, citr, state] = PSO_search(X, evals, slst, f, v, Pbestx, Pbestf, Gbestx, Gbestf, i, bound, citr, n,termination,nameR,state);
    end
end
P = Gbestx; index = find(state~=1); pf = Gbestf;
[~,b] = min(pf); 
Exactly = P;
Exactly(find(state~=1),:) = [];
end