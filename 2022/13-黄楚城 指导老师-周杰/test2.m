clc;
clear;


load AR2400n_2000d_120c.mat
load AR_noise.mat
X = X(1:800,:);

% load ORL_400n_1024d_40c.mat
% load ORL_noise.mat

% load YALE_165n_1024d_15c.mat
% load YALE_noise.mat
% 随机选取1/5样本添加噪声块 
randX = randsample(800,160,false);

noiseX = X_Noise(randX,:);
cleanX = X(randX,:);
X1 = X;
for i = 1:size(randX,1)
    tmp = randX(i);
    X(tmp,:) = X_Noise(tmp,:);
end

objPCA = zeros(1,5);
objTRPCA = zeros(1,5);
tmp = 1;

for dim_k = 10:10:50
    % TRPCA
    m = floor(0.8*size(X,1));    % 设置1/5为mild noise
    [iter,re,F,A,b] = trpca(X',dim_k,m);
    reNoise = norm((cleanX-(noiseX*F*F'+b')),2);
    objTRPCA(tmp) = reNoise;

    % PCA
    [mapping_data,mapping] = pca(X,dim_k);
    re1 = norm((cleanX-(noiseX*mapping.M*mapping.M'+mapping.mean)),'fro');
    objPCA(tmp) = re1;
    
    tmp = tmp+1;

end
x = 10:10:50;
plot(x,objTRPCA,'-*b',x,objPCA,'-or');
legend('T-RPCA','PCA','Location','east');
xlabel('Dimensions');
ylabel('Reconstruction error(occlusion)');

