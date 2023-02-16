function [X_train, X_test, Y_train, Y_test] = get_X_and_labels(data_mat, class_nums, num, train_nums, width, heigth)
    %---------构建X---------- 
    %X_train为d*N_train  X_test为d*N_test
    
    test_nums = num-train_nums;  %其余为测试集
    X_train=[];
    X_test = [];
    n=1;
    while n<=class_nums  
        rand_num = randperm(num, train_nums);  %从每个人的num张人脸随机抽取train_nums个作为训练样本
        for z =1:num
            x=[];
            for i=1:heigth
                for j =1:width
                    x=[x, data_mat(i,j,(n-1)*num+z)];
                end
            end
            if ismember(z, rand_num)
                X_train = [X_train, transpose(x)];
            else
                X_test = [X_test, transpose(x)];
            end

        end
        n = n+1; 
    end


    %------标签-------------
    Y_train = zeros(class_nums, class_nums*train_nums);
    Y_test = zeros(class_nums ,class_nums*test_nums);
    for i = 1:class_nums
        Y_train(i, (i-1)*train_nums+1:i*train_nums) = 1;
        Y_test(i, (i-1)*test_nums+1:i*test_nums) = 1;
    end
end