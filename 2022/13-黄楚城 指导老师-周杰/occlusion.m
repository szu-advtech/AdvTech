clc;clear;
% load AR2400n_2000d_120c.mat;
load YALE_165n_1024d_15c.mat
% load ORL_400n_1024d_40c.mat
% X=fea;
[n, d] = size(X);

csize = 15;
img_size_row = 32;
img_size_col = 32;
type = 1;
imgsX = [];
imgsX_Noise = [];
imgs010 = [];
imgs10 = [];
X_Noise = zeros(n,d);
for i =1:1:size(X,1)
    im  = X(i,:);
    im1  = reshape(im,img_size_row,img_size_col);
    imgs010 = [imgs010,im1];
    imgsX{i} = im1;
    im2  = im_noise_block(im1,csize,type);
    imgs10 = [imgs10,im2];
    imgsX_Noise{i} = im2;
    xi  = reshape(im2,1,img_size_row*img_size_col);
    X_Noise(i,:) = xi;
end
save('YALE_noise.mat','X_Noise','y');
imgs20 = [imgs010;imgs10];
imshow(imgs20,[10,250]);




