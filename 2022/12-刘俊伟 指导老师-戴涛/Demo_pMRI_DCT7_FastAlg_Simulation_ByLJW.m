% load ./data/data1s.mat
%%%%%%Read the data
close all;clear;clc;
fprintf('程序开始执行...\n');
%添加文件路径
AddFilePath;
%%%%%%%%%%%%%%读数据
ImgPath  =  './pMRI_FADHFA/Data/phantom.png';
ImgData  =  double(imread(ImgPath))/255;
[ImgRow, ImgCol,ImgDim] = size(ImgData);
MaskPath = './pMRI_FADHFA/Data/CartesianMask256_0_33_2.png';
Mask = imread(MaskPath);  %Get a sampling  model of k-space data
Mask = imrotate(Mask,270);
sample_rate = sum(Mask(:)) / numel(Mask);
fprintf('采样率：%.2f%% \n', sample_rate*100);
fprintf('读取数据成功，开始利用FADHF算法重建图像...\n');
%%%%%%%%%%%Simulation Low Resolution Image%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Simulation True Sensitivity%%%%%%%%%%%
SenCoi = zeros(ImgRow,ImgCol,4);
Com =  complex(1,1)/sqrt(2);
for i = 1:ImgRow
    for j = 1:ImgCol
        SenCoi(i,j,1) =Com*25000 / (25000 + (i+40)^2 + (j+20)^2);
        SenCoi(i,j,2) = Com*25000 / (25000 + (i+50)^2 + (j-290)^2);
        SenCoi(i,j,3) = Com*25000 / (25000 + (i-290)^2 + (j+10)^2);
        SenCoi(i,j,4) = Com*25000 / (25000 + (i-280)^2 + (j-310)^2);
    end
end
SenCoi_SOS = sqrt(sum(abs(SenCoi).^(2),3));

%%%%%%Generate the noisy coil images
sigma = 0.01;
for i = 1:4
    Noi = sigma*complex(randn(ImgRow,ImgCol),randn(ImgRow,ImgCol));
    ImgCoi(:,:,i) = SenCoi(:,:,i).*ImgData+Noi;
%     figure;
%     imshow(abs(ImgCoi(:,:,i)),[]);
    ksfull(:,:,i) = fftshift(fft2(ImgCoi(:,:,i)));
end%End for "i=1:4"
% ksfull = NormalizedCoeByEnergy(ksfull);
%         load ./data/Simulation_phantom2.mat
for i=1:size(ksfull,3)
    ksData(:,:,i)  =  Mask.*ksfull(:,:,i);
end
for i=1:size(ksfull,3)
    ksLR(:,:,i) = ifft2(ifftshift(ksData(:,:,i)));
end
Clipsos = sqrt(sum(abs(ksLR).^(2),3));
% figure(97);
% imshow(abs(Clipsos),[])

%%%%%%%%%%%%%%%%%
%Load the filters, just use the five filters
load ./pMRI_FADHFA/Data/DCT7.mat; %FilOpe, TraFilOpe
Tem = FilOpe(:,:,1:5);
FilOpe = Tem;
Tem = TraFilOpe(:,:,1:5);
TraFilOpe = Tem;
%%Iteration initilization  Parameters setting
Itr_Max = 100;
Lev = 2;                               %Decomposed levels
Itr_w = ImgDCT7Dec2(abs(Clipsos),Lev,FilOpe);
Itr_u  = Itr_w;
L = max(SenCoi_SOS(:))^2;
% L= 1;
Alpha = 1;
Beta = 1/ Alpha - L/2-0.001;
Rho = min(1/ Alpha - L/2, 1/Beta)*(1-sqrt(Beta/(1/ Alpha - L/2))) ;
Gamma = (1+max(1/2, L/(L+2*Rho)))/(2*max(1/2, L/(L+2*Rho)))-1-eps;
IsFix = 0;
tk  = 1;





%% PD3O
fprintf('接下来开始利用PD3O算法重建图像...\n');
ksData = ksfull;%笔误，下面代码是另一个文件搬过来的，懒得修改变量名
% ksData = NormalizedCoeByEnergy(ksData);
% ground_truth = sos(IFFT2_3D_N(ksData));
% acquire undersampled data using ksfull and Mask 
[kx, ky, coils] = size(ksData);
Mask = repmat(Mask, [1 1 coils]);
un_ksdata = ksData .* Mask;
un_ksdata = NormalizedCoeByEnergy(un_ksdata);

% estimate initial sensitivity from undersampled ksData
% sense_map = get_sensitivity(ksData, Mask);
sense_map = SenCoi;

% prepare parameters for PD3O algorithm
maxiter = 50;
level = 2;%紧框架分解层数
g = un_ksdata;
lambda = 0.0005;
uk = sos(IFFT2_3D_N(un_ksdata));
M = SMatrix(Mask, sense_map);
[x1] = PD3OMRI(g, Mask, ...
              sense_map,level,...
              'lambda', lambda, ...
              'method', '2DTF', ...
              'maxiter', maxiter, ...
              'upsense', false, ...
              'verbose', true);
%%%%%%%%% show zoom-in area
region = imcrop(x1,position);
region = imresize(region,8,'nearest');
f5=figure;
movegui(f5,[1200,200]);
imshow(region,[min(x1(:)),max(x1(:))]);
title('PD3O重建图的细节部分放大')
%%%%%%%% compute NMSE
x1Normalized = (x1-min(min(x1))*ones(ImgRow,ImgCol))./(max(max(x1))-min(min(x1)));
Err_PD3O = x1Normalized - ImgDataNormalized;
NMSE_PD3O = sum(Err_PD3O(:).^2)/sum(ImgDataNormalized(:).^2);

