function [train_output] = func_digahole(train_data,holesize)
%% Introduction
% Ϊ��̽��³�����㷨�����ܣ���ѵ����������һ������������Ĵ�С����ѡ��Ϊ 5x5
% 10x10 15x15 20x20������ÿ�ζ��������
% Input: train_data ѵ�����ݼ�
%        holesize ��ѡ��Ϊ5 10 15 20
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