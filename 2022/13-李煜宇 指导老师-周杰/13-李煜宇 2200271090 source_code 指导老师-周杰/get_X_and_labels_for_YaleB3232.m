function [X_train, X_test, train_labels, test_labels] = get_X_and_labels_for_YaleB3232(X, y, class_nums, train_nums)
    %---------构建X---------- 
    %X_train为d*N_train  X_test为d*N_test
    
    n = size(y, 1);

    X_train=[];
    X_test = [];
    
   % Y = zeros(n, class_nums);
    Y = zeros(class_nums, n);
    for k = 1:n
        Y(y(k), k) = 1;
    end
    train_labels = [];%zeros(class_nums*train_nums ,class_nums);
    test_labels = [];%zeros(n-class_nums*test_nums ,class_nums);
    
    
    i = 1;
    index = 0;
    while i<= class_nums
        if i==11 || i==13
            rand_num = randperm(60, train_nums);
            for j = 1:60          
                if ismember(j, rand_num)
                	X_train = [X_train, X(:, index+j)];
                    train_labels = [train_labels, Y(:, index+j)];
                else
                    X_test = [X_test, X(:, index+j)];
                    test_labels = [test_labels, Y(:, index+j)];
                end
            end
            index = index+60;
            
        elseif i==12
            rand_num = randperm(59, train_nums);
            for j = 1:59
                
                if ismember(j, rand_num)
                	X_train = [X_train, X(:, index+j)];
                    train_labels = [train_labels, Y(:, index+j)];
                else
                    X_test = [X_test, X(:, index+j)];
                    test_labels = [test_labels, Y(:, index+j)];
                end
            end
            index = index+59;
            
        elseif i==15
            rand_num = randperm(62, train_nums);
            for j = 1:62
                if ismember(j, rand_num)
                	X_train = [X_train, X(:, index+j)];
                    train_labels = [train_labels, Y(:, index+j)];
                else
                    X_test = [X_test, X(:, index+j)];
                    test_labels = [test_labels, Y(:, index+j)];
                end
            end
            index = index+62;
        elseif i==14 || i==16 || i==17
            rand_num = randperm(63, train_nums);
            for j = 1:63
                if ismember(j, rand_num)
                	X_train = [X_train, X(:, index+j)];
                    train_labels = [train_labels, Y(:, index+j)];
                else
                    X_test = [X_test, X(:, index+j)];
                    test_labels = [test_labels, Y(:, index+j)];
                end
            end 
            index = index+63;
        else
            rand_num = randperm(64, train_nums);        
            for j = 1:64        
                if ismember(j, rand_num)
                	X_train = [X_train, X(:, index+j)];
                    train_labels = [train_labels, Y(:, index+j)];
                else
                    
                    X_test = [X_test, X(:, index+j)];
                    test_labels = [test_labels, Y(:, index+j)];
                end
            end
            index = index+64;
        end
      
        i = i+1;
    end
    
    train_labels = train_labels';
    test_labels = test_labels';
    
end