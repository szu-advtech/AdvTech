

% 输入训练数据、训练标签、测试数据、训练标签、SVM参数cmd
% 以结构体的形式：返回分类准确度、召回率等指标

function svrData = my_getSvrData(trainData,trainScoresA,testData,testScoresA,cmd)


    A_SVR_MMR = svmtrain(trainScoresA, trainData, cmd);

    % Apre_MMR 是预测的值，Aacc的第二个参数是均方误差
    % Aacc= sum(T.*T)/n  其中T是误差,n是样本数，与均方根误差=均方误差开根号
    % Aacc：a vector with accuracy, 
    %       mean squared error-均方误差, 
    %       squared correlation coefficient-平方相关系数
    % Adec：If selected, probability estimate vector 

    % Adec 直接看成scores，SVM该项也是一样的，直接看成scores
    [ ~, Aacc, Adec ] = svmpredict(testScoresA, testData, A_SVR_MMR);
    if Aacc(1,1)==0 % 如果准确率为0，那么... 这里的意思是？ 答：把平方相关系数的根号，作为准确率
        % Adec和Aacc的非数值 非实数 无穷大 ---> eps
        Adec(isinf(Adec)) = eps;  
        Adec(isnan(Adec)) = eps; 
        Adec(~isreal(Adec)) = eps;
        Aacc(isinf(Aacc)) = eps;   
        Aacc(isnan(Aacc)) = eps;  
        Aacc(~isreal(Aacc)) = eps;
        svrData.ccSqrt=sqrt(Aacc(3,1)); % 看成SVM中的准确率
 
    else % 如果准确率不为0，直接将acc=0.5
        svrData.ccSqrt=0.5; % 看成SVM中的准确率
    end
    
    svrData.Adec=Adec;              % Adec好像是与Apre_MMR相等的，即它等于预测的值  
    svrData.RMSE=sqrt(Aacc(2,1));	% 均方根误差，sqrt(均方误差)
    svrData.ccOrg=Aacc(3,1);
    svrData.Atest=testScoresA;

end