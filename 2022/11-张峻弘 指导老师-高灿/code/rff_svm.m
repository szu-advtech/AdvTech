function model = rff_svm(Xtrain, Ytrain, param, varargin)
%rff_svm: SVM with random fourier features. 
%    Input:
%       Xtrain: training data, each row is a sample.
%       Ytrain: label of data, a col vector.
%       param:  the hyper-parameters.
%       options (varargin):
%              FeatureType: could be RFF/RM-BCSIK/Optimized. Corresponds to
%                           different type of random features.
%                     sign: could be true/false. Whether generate binary
%                           code.
%    Output: learned model.
%
%    Written by Junhong Zhang, 2022.11.10.

p = inputParser;
p.KeepUnmatched(true);
p.addParameter("FeatureType", "Optimized", ...
    @(x) assert(ismember(lower(x), ["rff", "rm-bcsik", "optimized"]), ...
               "FeatureType should be RFF/RM-BCSIK/Optimized."));
p.addParameter("sign", true, ...
    @(x) assert(islogical(x), ...
               "The sign flag should be true or false."));
parse(p, varargin{:});

opt = p.Results;
switch lower(opt.FeatureType)
    case "rff"
        W = randn(size(Xtrain,2), param.dim) / param.sigma;
        b = rand(1, param.dim) * 2*pi;
        Ztrain = makeRFF(W, b, Xtrain');
        D = param.dim;
        model.name = "rf-ff";
    case "rm-bcsik"
        m = size(Xtrain, 2);
        [W, b] = rmbcsik(m, param.sigma, param.dim);
        W = W(1:m, :);                   % zero paddings
        Ztrain = makeRFF(W, b, Xtrain');
        model.name = "rf-rmbcsik";
        D = param.dim;
    case "optimized"
        [W, b, alpha, alpha_distrib] = optimizeGaussianKernel(Xtrain', Ytrain, ...
            param.dim, param.rho, param.tol, param.sigma);
        D = length(alpha);
        [D, W, b] = createOptimizedGaussianKernelParams(D, W, b, alpha_distrib);
        Ztrain = makeRFF(W, b, Xtrain');
        model.name = "rf-optimized";
end

if opt.sign
    Ztrain = sign(Ztrain);
end
Ztrain = Ztrain';

svm = liblineartrain(Ytrain, sparse(Ztrain), '-s 3 -q');
model.svm = svm;
model.rffW = W;
model.rffb = b;
model.rffD = D;

end