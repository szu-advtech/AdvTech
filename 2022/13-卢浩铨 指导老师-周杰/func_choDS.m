function [train_data,train_label,test_data,test_label,data_d,data_n] = func_choDS(apart_r,varargin)
%% 函数获取数据集
% 通过给定的文件路径和相应的数据库名称，加载相应的数据集
% 默认的文件夹是当前路径，默认的数据集是ORL数据集
% 默认的划分参数是0.7
% 输出训练集、训练集标签、测试集、测试集标签、样本维度、样本总数
% 输出的训练矩阵的维度是 d * n
    defaultapart_r = 0.7;
    defaultpath = './';
    defaultsign = 'ORL';
    p = inputParser;
    addOptional(p,'apart_r',defaultapart_r);
    addParameter(p,'path',defaultpath);
    addParameter(p,'sign',defaultsign);
    parse(p,apart_r,varargin{:});
    
    apart_parm = p.Results.apart_r;
    file_path = p.Results.path;
    package_sign = p.Results.sign;
    
    %% AR数据集
%     总共有2400张人脸图像，其中分为120类，每一类都按顺序存储
%     则一类有20个样本，每个样本的维度为 50 * 40
%     存放的方式为 50*40*2400
    if strcmp(package_sign,'AR')
%         load data
        data = load(strcat(file_path,'AR120p20s50by40.mat'));
        data = data.AR120p20s50by40;
        data = reshape(data,[2000 2400]);
        [data_d, data_n] = size(data);
        
%         make label 
        label = zeros(1,data_n);
        for i = 1:data_n
            label(i) = floor((i - 1) / 20) + 1;
        end
        
%         train-test data
        train_data = [];
        train_label = [];
        test_data = [];
        test_label = [];
        
        totl_ind = 1:20;
        for i = 1:120
            part_ind = randperm(20,20 * apart_parm);
            train_data = [train_data,data(:,part_ind + (i - 1) * 20)];
            train_label = [train_label,label(:,part_ind + (i - 1) * 20)];
            not_part_ind = setdiff(totl_ind,part_ind);
            test_data = [test_data,data(:,not_part_ind + (i - 1) * 20)];
            test_label = [test_label,label(:,not_part_ind + (i - 1) * 20)];
        end
        
    %% Binary数据集
%     手写数字及大写字母的数据集
%     则一共有36个类。其中每个类都有39个样本
%     每个样本都是 20*16 的图片
    elseif strcmp(package_sign,'Binary')
        database = load(strcat(file_path,'binaryalphadigs.mat'));
        binary = database.Binary;
        data = reshape(binary,[320 1404]);
        label = database.label;
        [data_d,data_n] = size(data);
        
        %% train_test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];
        
        for i = 1:36
            ind_list = find(label==i);                        % 满足该类的下标（在数据中的位置）
            totl_ind = 1:39;                                  % 该类下标表的总下标  
            part_ind = randperm(39,floor(39 * apart_parm));          % 取出该类下标中的一部分作为训练集的下标表的下标
            not_part_ind = setdiff(totl_ind,part_ind);        % 下标表中作为测试集的表的下标
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end
        
    %% COIL100数据集
%     不同角度物品的数据集
%     一共有7200个样本，每类样本有72个
%     数据矩阵为 1024*7200
%     一共有100个类
    elseif strcmp(package_sign,'COIL100')
        database = load(strcat(file_path,'COIL100.mat'));
        data = database.COIL100;
        label = database.gnd;
        label = label';
        [data_d,data_n] = size(data);
        
        %% train_test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];
        
        for i = 1:100
            ind_list = find(label==i);                        % 满足该类的下标（在数据中的位置）
            totl_ind = 1:72;                                  % 该类下标表的总下标  
            part_ind = randperm(72,floor(72 * apart_parm));          % 取出该类下标中的一部分作为训练集的下标表的下标
            not_part_ind = setdiff(totl_ind,part_ind);        % 下标表中作为测试集的表的下标
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end
        
    %% FERET数据集
%     一共1400张图片，每张图片大小为 40*40
%     其中划分为200个类，每个类有7张图，按顺序存放
    elseif strcmp(package_sign,'FERET')
        data = load(strcat(file_path,'FERET74040.mat'));
        data = data.FERET74040;
        data = reshape(data,[1600,1400]);
        [data_d,data_n] = size(data);

        %% Labels of all samples
        % [1,...,1,2,...,2,...]
        label = zeros(1,1400);
        for i = 1:1400
            label(i) = floor((i - 1) / 7) + 1;
        end

        %% train-test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];

        totl_ind = 1:7;
        for i = 1:200
            part_ind = randperm(7,round(7 * apart_parm));
            train_data = [train_data,data(:,part_ind + (i - 1) * 7)];
            train_label = [train_label,label(:,part_ind + (i - 1) * 7)];
            not_part_ind = setdiff(totl_ind,part_ind);
            test_data = [test_data,data(:,not_part_ind + (i - 1) * 7)];
            test_label = [test_label,label(:,not_part_ind + (i - 1) * 7)];
        end
        
    %% Pose数据集   
%     一共1632张图片，每张图片大小为 32*32
%     其中划分为68个类，每个类有24张图，按顺序存放
    elseif strcmp(package_sign,'Pose')
        data = load(strcat(file_path,'Pose09_32by32.mat'));
        data = data.Pose09_32by32;
        data = data';
        [data_d,data_n] = size(data);

        %% Labels of all samples
        % [1,...,1,2,...,2,...]
        label = zeros(1,1632);
        for i = 1:1632
            label(i) = floor((i - 1) / 24) + 1;
        end

        %% train-test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];

        totl_ind = 1:24;
        for i = 1:68
            part_ind = randperm(24,round(24 * apart_parm));
            train_data = [train_data,data(:,part_ind + (i - 1) * 24)];
            train_label = [train_label,label(:,part_ind + (i - 1) * 24)];
            not_part_ind = setdiff(totl_ind,part_ind);
            test_data = [test_data,data(:,not_part_ind + (i - 1) * 24)];
            test_label = [test_label,label(:,not_part_ind + (i - 1) * 24)];
        end
        
    %% Yale数据集   
%     一共165张图片，每张图片大小为 50*40
%     其中划分为15个类，每个类有11张图，按顺序存放
    elseif strcmp(package_sign,'Yale')
        data = load(strcat(file_path,'Yale5040165.mat'));
        data = data.Yale5040165;
        data = reshape(data,[2000 165]);
        [data_d,data_n] = size(data);

        %% Labels of all samples
        % [1,...,1,2,...,2,...]
        label = zeros(1,165);
        for i = 1:165
            label(i) = floor((i - 1) / 11) + 1;
        end

        %% train-test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];

        totl_ind = 1:11;
        for i = 1:15
            part_ind = randperm(11,round(11 * apart_parm));
            train_data = [train_data,data(:,part_ind + (i - 1) * 11)];
            train_label = [train_label,label(:,part_ind + (i - 1) * 11)];
            not_part_ind = setdiff(totl_ind,part_ind);
            test_data = [test_data,data(:,not_part_ind + (i - 1) * 11)];
            test_label = [test_label,label(:,not_part_ind + (i - 1) * 11)];
        end      
        
    %% YaleB数据集
%     人脸的数据集
%     一共有2414个样本，每类样本数量不定，大多在64个
%     单个样本数据维度为32*32
%     一共有38个类
    elseif strcmp(package_sign,'YaleB')
        database = load(strcat(file_path,'YaleB_32x32.mat'));
        data = database.fea;
        label = database.gnd;
        data = data';
        label = label';
        [data_d,data_n] = size(data);
        che_laIn = tabulate(label);
        
        %% train_test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];
        
        for i = 1:38
            ind_list = find(label==i);                        % 满足该类的下标（在数据中的位置）
            totl_ind = 1:che_laIn(i,2);                                  % 该类下标表的总下标  
            part_ind = randperm(che_laIn(i,2),floor(che_laIn(i,2) * apart_parm));          % 取出该类下标中的一部分作为训练集的下标表的下标
            not_part_ind = setdiff(totl_ind,part_ind);        % 下标表中作为测试集的表的下标
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end
        
    %% 若以上几种情况都不是，则默认为输出的是ORL数据集
    %% ORL数据集
%     总共有400张人脸图像
%     初始存放于一个结构体中
%     其中数据的名称为ORL4646，数据的存放方式为 46 * 46 * 400
%     意味着单个人脸样本的维度为2116（列向量）
%     按顺序一共分为40个类
%     每个类有十个人脸
    else
        %% ORL data set
        data = load(strcat(file_path,'ORL4646.mat'));
        data = data.ORL4646;
        data = reshape(data,[46*46,400]);
        [data_d,data_n] = size(data);

        %% Labels of all samples
        % [1,...,1,2,...,2,...]
        label = zeros(1,400);
        for i = 1:400
            label(i) = floor((i - 1) / 10) + 1;
        end

        %% train-test data
        train_data = [];
        test_data = [];
        train_label = [];
        test_label = [];

        totl_ind = 1:10;
        for i = 1:40
            part_ind = randperm(10,10 * apart_parm);
            train_data = [train_data,data(:,part_ind + (i - 1) * 10)];
            train_label = [train_label,label(:,part_ind + (i - 1) * 10)];
            not_part_ind = setdiff(totl_ind,part_ind);
            test_data = [test_data,data(:,not_part_ind + (i - 1) * 10)];
            test_label = [test_label,label(:,not_part_ind + (i - 1) * 10)];
        end
    end
end