function [T,Z]=SELF(X,Y,beta,r,metric,kNN)
%
% SELF: Semi-Supervised Local Fisher Discriminant Analysis
%       for Dimensionality Reduction
%
% Usage:
%       [T,Z]=SELF(X,Y,beta,r,metric,kNN)
%
% Input:
%    X:      d x n matrix of original samples
%            d --- dimensionality of original samples
%            n --- the number of samples 
%    Y:      n dimensional vertical vector of class labels
%            (each element takes an integer between 0 and c,
%            where c is the number of classes)
%            0:             unlabeled 
%            {1,2, ... ,c}: labeled
%    beta:   degree of semi-supervisedness (0 <= beta <= 1; default is 0.5 )
%            0: totally supervised (discard all unlabeled samples)
%            1: totally unsupervised (discard all label information)
%    r:      dimensionality of reduced space (default is d)
%    metric: type of metric in the embedding space (default is 'weighted')
%            'weighted'        --- weighted eigenvectors 
%            'orthonormalized' --- orthonormalized
%            'plain'           --- raw eigenvectors
%    kNN:    affinity parameter used in local scaling heuristic (default is 7)
%
% Output:
%    T: d x r transformation matrix (Z=T'*X)
%    Z: r x n matrix of dimensionality reduced samples 
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/SELF/

if nargin<2
  error('Not enough input arguments.')
end
[d n]=size(X);

if nargin<3
  beta=0.5;
end

if nargin<4
  r=d;
end

if nargin<5
  metric='weighted';
end

if nargin<6
  kNN=7;
end

flag_label=(Y~=0);
nlabel=sum(flag_label);

X2=sum(X.^2,1);
dist2=repmat(X2(flag_label),n,1)+repmat(X2',1,nlabel)-2*X'*X(:,flag_label);
[sorted,index]=sort(dist2,1);
kNNdist2=max(sorted(kNN+1,:),0);
localscale=sqrt(kNNdist2);
LocalScale=localscale'*localscale;
flag=(LocalScale>0);
A=zeros(nlabel,nlabel);
dist2tmp=dist2(flag_label,:);
A(flag)=exp(-dist2tmp(flag)./LocalScale(flag));

Wlb=zeros(nlabel,nlabel);
Wlw=zeros(nlabel,nlabel);
Ylabel=Y(flag_label);
for class=1:max(Y)
  flag_class=(Ylabel==class);
  nclass=sum(flag_class);
  if nclass~=0
    tmp =flag_class*1;
    tmp2=(~flag_class)*1;
    Wlb=Wlb+A*(1/nlabel-1/nclass).*(tmp'*tmp)+(tmp'*tmp2)/nlabel;
    Wlw=Wlw+(A/nclass).*(tmp'*tmp);
  end
end

Slb=X(:,flag_label)*(diag(sum(Wlb,1))-Wlb)*X(:,flag_label)';
Slw=X(:,flag_label)*(diag(sum(Wlw,1))-Wlw)*X(:,flag_label)';

Srlb=(1-beta)*Slb+beta*cov(X',1);
Srlw=(1-beta)*Slw+beta*eye(d);

Srlb=(Srlb+Srlb')/2;
Srlw=(Srlw+Srlw')/2;

if r==d
  [eigvec,eigval_matrix]=eig(Srlb,Srlw);
else
  opts.disp = 0; 
  [eigvec,eigval_matrix]=eigs(Srlb,Srlw,r,'la',opts);
end
eigval=diag(eigval_matrix);
[sort_eigval,sort_eigval_index]=sort(eigval);
T0=eigvec(:,sort_eigval_index(end:-1:1));

switch metric %determine the metric in the embedding space
  case 'weighted'
   T=T0.*repmat(sqrt(abs(sort_eigval(end:-1:1)))',[d,1]);
  case 'orthonormalized'
   [T,dummy]=qr(T0,0);
  case 'plain'
   T=T0;
end

Z=T'*X;

