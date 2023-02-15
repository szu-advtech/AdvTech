function  [dmodel, perf] = dacefit_3(S, Y, regr, corr, theta0)
%DACEFIT Constrained non-linear least-squares fit of a given correlation
% model to the provided data set and regression model
%
% Call
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0)
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
%
% Input
% S, Y    : Data points (S(i,:), Y(i,:)), i = 1,...,m
% regr    : Function handle to a regression model
% corr    : Function handle to a correlation function
% theta0  : Initial guess on theta, the correlation function parameters
% lob,upb : If present, then lower and upper bounds on theta
%           Otherwise, theta0 is used for theta
%
% Output
% dmodel  : DACE model: a struct with the elements
%    regr   : function handle to the regression model
%    corr   : function handle to the correlation function
%    theta  : correlation function parameters
%    beta   : generalized least squares estimate
%    gamma  : correlation factors
%    sigma2 : maximum likelihood estimate of the process variance
%    S      : scaled design sites
%    Ssc    : scaling factors for design arguments
%    Ysc    : scaling factors for design ordinates
%    C      : Cholesky factor of correlation matrix
%    Ft     : Decorrelated regression matrix
%    G      : From QR factorization: Ft = Q*G' .
% perf    : struct with performance information. Elements
%    nv     : Number of evaluations of objective function
%    perf   : (q+2)*nv array, where q is the number of elements 
%             in theta, and the columns hold current values of
%                 [theta;  psi(theta);  type]
%             |type| = 1, 2 or 3, indicate 'start', 'explore' or 'move'
%             A negative value for type indicates an uphill step

% hbn@imm.dtu.dk  
% Last update September 3, 2002

% Check design points
[m n] = size(S);  % number of design sites and their dimension%%%%%%m为点的数目，n为维度
sY = size(Y);
if  min(sY) == 1%%%%%%%min(sY) 表示变量的维度
    Y = Y(:);   lY = max(sY);  sY = size(Y);%%%%%%%Y为响应，lY为点的数目
else
    lY = sY(1); 
end
if m ~= lY
  error('S and Y must have the same number of rows')
end

if  any(theta0 <= 0)
    error('theta0 must be strictly positive')
end

% Normalize data
mS = mean(S);   sS = std(S);%%%%%%%%%%%%mean，std表示的为每一列的均值和标准差
mY = mean(Y);   sY = std(Y);
% 02.08.27: Check for 'missing dimension'
j = find(sS == 0);
if  ~isempty(j)
    sS(j) = 1; 
end
j = find(sY == 0);
if  ~isempty(j)
    sY(j) = 1;
end
S = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);%%%%%%%repmat为复制矩阵函数，将基矩阵复制m行1列
Y = (Y - repmat(mY,m,1)) ./ repmat(sY,m,1);

% Calculate distances D between points
mzmax = m*(m-1) / 2;        % number of non-zero distances
ij = zeros(mzmax, 2);       % initialize matrix with indices
D = zeros(mzmax, n);        % initialize matrix with distances有mzmax个非零距离，但是是以每一维度的距离为计算统计的
ll = 0;
for k = 1 : m-1
  ll = ll(end) + (1 : m-k);
  ij(ll,:) = [repmat(k, m-k, 1) (k+1 : m)']; % indices for sparse matrix，ij为稀疏矩阵的非零元素的行标和列标
  D(ll,:) = repmat(S(k,:), m-k, 1) - S(k+1:m,:); % differences between points第一次迭代，D的1至m-1行，第一个点与其它m-1个点将的距离，第二次迭代，m至2m-3行，为第二个点与其它们m-2个点的距离
end
if  min(sum(abs(D),2) ) == 0
  error('Multiple design sites are not allowed')
end
% Regression matrix
F = feval(regr, S);  [mF p] = size(F);%%%%%%%%%%%%%%%%%%%%%%检查的点的数目是否满足回归条件F为回归基函数矩阵
if  mF ~= m, error('number of rows in  F  and  S  do not match'), end
if  p > mF,  error('least squares problem is underdetermined'), end

% parameters for objective function
par = struct('corr',corr, 'regr',regr, 'y',Y, 'F',F, ...
  'D', D, 'ij',ij, 'scS',sS);

% Determine theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%通过GA算法优化theta值，它是一个行向量
global ppar
ppar=par;
FTT=10^3;
Xtheta=10;
for ii=1:1
%     ii
    lb=0.01;
    ub=20;
    Range=[lb,ub];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%获取初始theta值
    bounds = Range;
    opts.maxevals=100;
    opts.maxits=2000;
    opts.maxdeep=2000;
    opts.testflag=0;
    opts.showits=0;
    opts.globalmin =0;
    Problem.f = 'findtheta';
    [FTT1,Xtheta1] = Direct(Problem,bounds,opts);
    if FTT1<FTT
        Xtheta=Xtheta1;
        FTT=FTT1;
    end
end
theta0=Xtheta;
  % Given theta
  theta = theta0(:);   %%%%%%%%%%%%%%%%%%%%%统一无论theta是行向量还是列向量，都统一设置成列向量的形式。
  [f  fit] = objfunc(theta, par);
  perf = struct('perf',[theta; f; 1], 'nv',1);
%   if  isinf(f)
%     error('Bad point.  Try increasing theta0')
%   end

% Return values
dmodel = struct('regr',regr, 'corr',corr, 'theta',theta.', ...
  'beta',fit.beta, 'gamma',fit.gamma, 'sigma2',sY.^2.*fit.sigma2, ...
  'S',S, 'Ssc',[mS; sS], 'Ysc',[mY; sY], ...
  'C',fit.C, 'Ft',fit.Ft, 'G',fit.G);
end



