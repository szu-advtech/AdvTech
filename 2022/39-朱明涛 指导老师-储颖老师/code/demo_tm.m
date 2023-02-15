clear,clc;
%% Parameters
lambda1 = 0.3;
lambda2 = 0.01*lambda1;
lambda3 = 0.1;

alpha = 0.8;

gamma = 2.2;
%% input file
hdr = hdrread("1.hdr");
% imwrite(hdr,"radianceMap_Stadium_Center.png");
[row,col,channel] = size(hdr);
tic;

%% Pretreatment
hdr_hsv = rgb2hsv(hdr);  % color transformation
hdr_v = hdr_hsv(:,:,channel);  %  luminance channel
hdr_v = log(hdr_v+0.0001);  %  converted to log domain
hdr_v = normalize(hdr_v);   %  归一化

%% Decomposition
[D1,D2,B2] = layer_decomp(hdr_v,lambda1,lambda2,lambda3);

%% processing
% figure,imshow(D1+0.4);
% imwrite(D1+0.4,'D1_by_L1L0.png');
sigma = max(D1(:));
D1 = R_func(D1,0,sigma,alpha,1);

% figure,imshow(D2+0.4);
% imwrite(D2+0.4,'D2_by_L1L0.png');
% figure,imshow(B2);
% imwrite(B2,'B2_by_L1L0.png');
B2= compress(B2,gamma,1);

hdr_vC = 0.8*B2 + D2 + 1.2*D1;

%% postprocessing 

hdr_vC = normalize(clampp(hdr_vC,0.005,0.995));
hdr_out = hsv2rgb(cat(3,hdr_hsv(:,:,1),hdr_hsv(:,:,2)*0.7,hdr_vC));
toc;
[Q, S, N, s_maps, s_local] = TMQI(hdr,im2uint8(hdr_out));
[Q,S,N]
figure,imshow(hdr_out);
% imwrite(hdr_out,"RecuL1L0_Stadium_Center.png");
