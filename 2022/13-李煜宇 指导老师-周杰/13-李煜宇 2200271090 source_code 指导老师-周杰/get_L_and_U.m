function [X, XL, XU, Y, YL, YU, YU_label] = get_L_and_U(X, Y, train_nums, labeled_nums, class_nums)
%划分XL和XU
%   此处显示详细说明
%
%
    %XL = zeros(size(X, 1), labeled_nums*class_nums);
    %XU = zeros(size(X, 1), (train_nums-labeled_nums)*class_nums);

    %YL = zeros(size(Y, 1), labeled_nums*class_nums);
    XL = [];
    XU = [];
    YL = [];
    YU = zeros(size(Y, 1), (train_nums-labeled_nums)*class_nums); 
    YU_label = [];

    n = 1;
    while n<=class_nums  
        rand_num = randperm(train_nums, labeled_nums); %从每个类中随机抽取labeled_nums个样本作为有标签
        for z =1:train_nums
            if ismember(z, rand_num)
                XL = [XL, X(:, (n-1)*train_nums+z)];
                YL = [YL, Y(:, (n-1)*train_nums+z)];
            else
                XU = [XU, X(:, (n-1)*train_nums+z)];
                YU_label = [YU_label, Y(:, (n-1)*train_nums+z)];
            end

        end
        n = n+1; 
    end
    
    X = [XL, XU];
    Y = [YL, YU];

end

