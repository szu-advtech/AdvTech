function [train_data,train_label,test_data,test_label,data_d,data_n] = func_choDS(apart_r,varargin)
%% ������ȡ���ݼ�
% ͨ���������ļ�·������Ӧ�����ݿ����ƣ�������Ӧ�����ݼ�
% Ĭ�ϵ��ļ����ǵ�ǰ·����Ĭ�ϵ����ݼ���ORL���ݼ�
% Ĭ�ϵĻ��ֲ�����0.7
% ���ѵ������ѵ������ǩ�����Լ������Լ���ǩ������ά�ȡ���������
% �����ѵ�������ά���� d * n
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
    
    %% AR���ݼ�
%     �ܹ���2400������ͼ�����з�Ϊ120�࣬ÿһ�඼��˳��洢
%     ��һ����20��������ÿ��������ά��Ϊ 50 * 40
%     ��ŵķ�ʽΪ 50*40*2400
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
        
    %% Binary���ݼ�
%     ��д���ּ���д��ĸ�����ݼ�
%     ��һ����36���ࡣ����ÿ���඼��39������
%     ÿ���������� 20*16 ��ͼƬ
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
            ind_list = find(label==i);                        % ���������±꣨�������е�λ�ã�
            totl_ind = 1:39;                                  % �����±������±�  
            part_ind = randperm(39,floor(39 * apart_parm));          % ȡ�������±��е�һ������Ϊѵ�������±����±�
            not_part_ind = setdiff(totl_ind,part_ind);        % �±������Ϊ���Լ��ı���±�
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end
        
    %% COIL100���ݼ�
%     ��ͬ�Ƕ���Ʒ�����ݼ�
%     һ����7200��������ÿ��������72��
%     ���ݾ���Ϊ 1024*7200
%     һ����100����
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
            ind_list = find(label==i);                        % ���������±꣨�������е�λ�ã�
            totl_ind = 1:72;                                  % �����±������±�  
            part_ind = randperm(72,floor(72 * apart_parm));          % ȡ�������±��е�һ������Ϊѵ�������±����±�
            not_part_ind = setdiff(totl_ind,part_ind);        % �±������Ϊ���Լ��ı���±�
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end
        
    %% FERET���ݼ�
%     һ��1400��ͼƬ��ÿ��ͼƬ��СΪ 40*40
%     ���л���Ϊ200���࣬ÿ������7��ͼ����˳����
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
        
    %% Pose���ݼ�   
%     һ��1632��ͼƬ��ÿ��ͼƬ��СΪ 32*32
%     ���л���Ϊ68���࣬ÿ������24��ͼ����˳����
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
        
    %% Yale���ݼ�   
%     һ��165��ͼƬ��ÿ��ͼƬ��СΪ 50*40
%     ���л���Ϊ15���࣬ÿ������11��ͼ����˳����
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
        
    %% YaleB���ݼ�
%     ���������ݼ�
%     һ����2414��������ÿ���������������������64��
%     ������������ά��Ϊ32*32
%     һ����38����
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
            ind_list = find(label==i);                        % ���������±꣨�������е�λ�ã�
            totl_ind = 1:che_laIn(i,2);                                  % �����±������±�  
            part_ind = randperm(che_laIn(i,2),floor(che_laIn(i,2) * apart_parm));          % ȡ�������±��е�һ������Ϊѵ�������±����±�
            not_part_ind = setdiff(totl_ind,part_ind);        % �±������Ϊ���Լ��ı���±�
            
            train_data = [train_data,data(:,ind_list(part_ind))];
            train_label = [train_label,label(:,ind_list(part_ind))];
            test_data = [test_data,data(:,ind_list(not_part_ind))];
            test_label = [test_label,label(:,ind_list(not_part_ind))];
        end
        
    %% �����ϼ�����������ǣ���Ĭ��Ϊ�������ORL���ݼ�
    %% ORL���ݼ�
%     �ܹ���400������ͼ��
%     ��ʼ�����һ���ṹ����
%     �������ݵ�����ΪORL4646�����ݵĴ�ŷ�ʽΪ 46 * 46 * 400
%     ��ζ�ŵ�������������ά��Ϊ2116����������
%     ��˳��һ����Ϊ40����
%     ÿ������ʮ������
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