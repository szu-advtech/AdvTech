clc;
clear;

load ORL_400n_1024d_40c.mat
load ORL_noise.mat

% 随机选取1/5样本添加噪声块 
randX = randsample(400,80,false);
noiseX = X_Noise(randX,:);
cleanX = X(randX,:);
for i = 1:size(randX,1)
    tmp = randX(i);
    X(tmp,:) = X_Noise(tmp,:);
end
% X = X(1:800,:);

% X1 = X_Noise(1:150,:);    % 选取1/5的样本数量
% X1 = [X1;X(150:750,:)];
dim_k = 20;      % 降维数目

% TRPCA
m = floor(0.8*size(X,1));    % 设置1/5为mild noise
[iter,re,F,A,b] = trpca(X',dim_k,m);
TRPCAY = (X-ones(size(X,1),1)*b')*F*F'+ones(size(X,1),1)*b';      %重构
TRPCAX = TRPCAY(randX,:);

% PCA
[mapping_data,mapping] = pca(X,dim_k);
PCAY = mapping_data * (mapping.M)' + mapping.mean;
PCAX = PCAY(randX,:);

csize = 15;
img_size_row = 32;
img_size_col = 32;
type = 4;
imgsX = [];
imgsX_Noise = [];

imgsX_10 = [];
imgsX_Noise10 = [];
imgsPCAX = [];
imgsTRPCAX = [];

for i = 2:1:size(X,2)
    im  = cleanX(i,:);
    im1  = reshape(im,img_size_row,img_size_col);
    imgsX_10 = [imgsX_10,im1];

    Noiseim  = noiseX(i,:);
    Noiseim1  = reshape(Noiseim,img_size_row,img_size_col);
    imgsX_Noise10 = [imgsX_Noise10,Noiseim1];

    PCAim = PCAX(i,:);
    PCAim1  = reshape(PCAim,img_size_row,img_size_col);
    imgsPCAX = [imgsPCAX,PCAim1];

    TRPCAim = TRPCAX(i,:);
    TRPCAim1  = reshape(TRPCAim,img_size_row,img_size_col);
    imgsTRPCAX = [imgsTRPCAX,TRPCAim1];

end

imgs20 = [imgsX_10;imgsX_Noise10;imgsPCAX;imgsTRPCAX];
imshow(imgs20,[10,250]);
