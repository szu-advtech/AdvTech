

% ����ѵ�����ݡ�ѵ����ǩ���������ݡ�ѵ����ǩ��SVM����cmd
% �Խṹ�����ʽ�����ط���׼ȷ�ȡ��ٻ��ʵ�ָ��

function svrData = my_getSvrData(trainData,trainScoresA,testData,testScoresA,cmd)


    A_SVR_MMR = svmtrain(trainScoresA, trainData, cmd);

    % Apre_MMR ��Ԥ���ֵ��Aacc�ĵڶ��������Ǿ������
    % Aacc= sum(T.*T)/n  ����T�����,n��������������������=����������
    % Aacc��a vector with accuracy, 
    %       mean squared error-�������, 
    %       squared correlation coefficient-ƽ�����ϵ��
    % Adec��If selected, probability estimate vector 

    % Adec ֱ�ӿ���scores��SVM����Ҳ��һ���ģ�ֱ�ӿ���scores
    [ ~, Aacc, Adec ] = svmpredict(testScoresA, testData, A_SVR_MMR);
    if Aacc(1,1)==0 % ���׼ȷ��Ϊ0����ô... �������˼�ǣ� �𣺰�ƽ�����ϵ���ĸ��ţ���Ϊ׼ȷ��
        % Adec��Aacc�ķ���ֵ ��ʵ�� ����� ---> eps
        Adec(isinf(Adec)) = eps;  
        Adec(isnan(Adec)) = eps; 
        Adec(~isreal(Adec)) = eps;
        Aacc(isinf(Aacc)) = eps;   
        Aacc(isnan(Aacc)) = eps;  
        Aacc(~isreal(Aacc)) = eps;
        svrData.ccSqrt=sqrt(Aacc(3,1)); % ����SVM�е�׼ȷ��
 
    else % ���׼ȷ�ʲ�Ϊ0��ֱ�ӽ�acc=0.5
        svrData.ccSqrt=0.5; % ����SVM�е�׼ȷ��
    end
    
    svrData.Adec=Adec;              % Adec��������Apre_MMR��ȵģ���������Ԥ���ֵ  
    svrData.RMSE=sqrt(Aacc(2,1));	% ��������sqrt(�������)
    svrData.ccOrg=Aacc(3,1);
    svrData.Atest=testScoresA;

end