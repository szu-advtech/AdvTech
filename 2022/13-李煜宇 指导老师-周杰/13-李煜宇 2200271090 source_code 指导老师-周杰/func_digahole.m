function [train_output] = func_digahole(train_data,holesize)
%% Introduction
% 为了探究鲁棒性算法的性能，将训练集数据挖一个洞，这个洞的大小可以选择为 5x5
% 10x10 15x15 20x20，并且每次都是随机的
% Input: train_data 训练数据集
%        holesize 可选择为5 10 15 20
% Output: train_output

%% Function body
[train_h,train_w,train_n] = size(train_data);
switch(holesize)
    case {5}
        train_output = zeros(train_h,train_w,train_n);
        z_mat = zeros(holesize,holesize);
        for i = 1:train_n
            x = randi([holesize,train_h - holesize]);
            y = randi([holesize,train_w - holesize]);
            train_output(:,:,i) = train_data(:,:,i);
            train_output(x:x+holesize-1,y:y+holesize-1,i) = z_mat;
        end
    case {10}
        train_output = zeros(train_h,train_w,train_n);
        z_mat = zeros(holesize,holesize);
        for i = 1:train_n
            x = randi([holesize,train_h - holesize]);
            y = randi([holesize,train_w - holesize]);
            train_output(:,:,i) = train_data(:,:,i);
            train_output(x:x+holesize-1,y:y+holesize-1,i) = z_mat;
        end
    case {15}
        train_output = zeros(train_h,train_w,train_n);
        z_mat = zeros(holesize,holesize);
        for i = 1:train_n
            x = randi(train_h - holesize);
            y = randi(train_w - holesize);
            train_output(:,:,i) = train_data(:,:,i);
            train_output(x:x+holesize-1,y:y+holesize-1,i) = z_mat;
        end
    case {20}
        train_output = zeros(train_h,train_w,train_n);
        z_mat = zeros(holesize,holesize);
        for i = 1:train_n
            x = randi(train_h - holesize);
            y = randi(train_w - holesize);
            train_output(:,:,i) = train_data(:,:,i);
            train_output(x:x+holesize-1,y:y+holesize-1,i) = z_mat;
        end
    otherwise
        train_output = zeros(train_h,train_w,train_n);
        z_mat = zeros(holesize,holesize);
        for i = 1:train_n
            x = randi([holesize,train_h - holesize]);
            y = randi([holesize,train_w - holesize]);
            train_output(:,:,i) = train_data(:,:,i);
            train_output(x:x+holesize-1,y:y+holesize-1,i) = z_mat;
        end
end