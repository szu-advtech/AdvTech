function [maxCc,ccRmse,ccDec,ccTest,ccBest_X,ccBest_Y,minRmse]=...
    my_findBestCc_SVR(matCc,matRmse,matDec,matTest)

    % �ҳ����cc(���ϵ��)�Լ���Ӧ��rmse��Ldec��Ltest��parg parc�����±�
    % ͬʱ�ҳ���С��rmse������������û�ж�Ӧ��ֵ


    minRmse=min(matRmse(:));

    maxCc=max(matCc(:)); % �������׼ȷ��.�������ظ���
    seqIndex=find(matCc==maxCc); % �ҳ������ظ������ֵ��˳������(����)
    [x,y]=ind2sub(size(matCc),seqIndex); % ������ת���ɾ��������,ע�� x,y������

    % ��Ȼ׼ȷ�����ֵ�������ظ�����ô�����ٿ� accArmse ָ��
    maxCc=[];
    ccRmse=[];
    ccDec=[];
    ccTest=[];
    ccBest_X=[];   % ��¼���ֵ��X����
    ccBest_Y=[];   % ��¼���ֵ��Y����
    
    for i=1:length(x)  % ����������׼ȷ�ȣ��������±��ҳ����ж�Ӧ������ָ��
        
        ccBest_X=[ccBest_X,x(i)];
        ccBest_Y=[ccBest_Y,y(i)];

        maxCc=[maxCc,matCc(x(i),y(i))];
        ccRmse=[ccRmse,matRmse(x(i),y(i))];
        ccDec=[ccDec,matDec(x(i),y(i))];
        ccTest=[ccTest,matTest(x(i),y(i))];
    end
    
    [~,maxMapIndex]=min(ccRmse(:)); % �ҳ���� accArmse.�������ظ���.������ظ��ľ�ѡ���һ�������ٽ�һ��������ָ����
    maxCc=maxCc(maxMapIndex);
    ccRmse=ccRmse(maxMapIndex);
    ccDec=ccDec{maxMapIndex};
    ccTest=ccTest{maxMapIndex};

    ccBest_X=ccBest_X(maxMapIndex);
    ccBest_Y=ccBest_Y(maxMapIndex);
end









