function [D_new, idx_new] = createOptimizedLinearKernelParams(D, idx_opt, alpha_distrib)
% CREATEOPTIMIZEDLINEARKERNELPARAMS generates parameters for an optimized
% linear kernel. If we ask to create more random features than the number
% of features in the optmized kernel, we will just return the number of
% features in the optimized kernel. If we ask for fewer, then we sample
% from the distribution for the optimized kernel.
%
% Inputs:
% D is the number of random features we wish to generate
% idx_opt, alpha_distrib are outputs from OptimizeLinearKernel
%
% Outputs:
% D_new the number of random features we actually have now
% idx_new the parameters for those random features
    idx_new = idx_opt;
    D_new = size(idx_opt,2);
    if D < D_new
        p = rand(D,1);
        inds=sum((repmat(p,1,length(alpha_distrib)) - repmat(alpha_distrib',D,1))>0,2)+1;
        idx_new = idx_opt(:,inds);
        D_new = D;
    end
end