function [D_new, W_new, b_new] = createOptimizedGaussianKernelParams(D, W_opt, b_opt, alpha_distrib)
% CREATEOPTIMIZEDGAUSSIANKERNELPARAMS generates parameters for an optimized
% Gaussian kernel. If we ask to create more random features than the number
% of features in the optmized kernel, we will just return the number of
% features in the optimized kernel. If we ask for fewer, then we sample
% from the distribution for the optimized kernel.
%
% Inputs:
% D is the number of random features we wish to generate
% W_opt, b_opt, alpha_distrib are outputs from OptimizeGaussianKernel
%
% Outputs:
% D_new the number of random features we actually have now
% W_new, b_new the parameters for those random features

    W_new = W_opt;
    b_new = b_opt;
    D_new = size(b_opt,2);
    if D < D_new
        p = rand(D,1);
        inds=sum((repmat(p,1,length(alpha_distrib)) - repmat(alpha_distrib',D,1))>0, 2) + 1;
        W_new = W_opt(:,inds);
        b_new = b_opt(inds);
        D_new = D;
    end
end