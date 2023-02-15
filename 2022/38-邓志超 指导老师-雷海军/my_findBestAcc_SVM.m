function [maxAcc,accSen,accSpec,accPrec,accAuc,accFscore,accLdec,accLtest,accTP,accFP,accBest_X,accBest_Y,maxAuc,aucLdec,aucLtest,aucTP,aucFP,aucBest_X,aucBest_Y]=...
    my_findBestAcc_SVM(matAcc,matSen,matSpec,matPrec,matAuc,matFscore,matLdec,matLtest,matTP,matFP)

    % -------------------- AUC -----------------------------
    [maxAuc,maxAucIndex]=max(matAuc(:)); % �������fscores.�������ظ���
    [aucBest_X,aucBest_Y]=ind2sub(size(matAuc),maxAucIndex); % ������ת���ɾ��������,ע�� x,y������
    aucLdec=matLdec{aucBest_X,aucBest_Y};
    aucLtest=matLtest{aucBest_X,aucBest_Y};
    aucTP=matTP{aucBest_X,aucBest_Y};
    aucFP=matFP{aucBest_X,aucBest_Y};
    
    
    % -------------------- Acc  -----------------------------
    maxAcc=max(matAcc(:)); % �������fscores.�������ظ���
    seqIndex=find(matAcc==maxAcc); % �ҳ������ظ������ֵ��˳������(����)
    [x,y]=ind2sub(size(matAcc),seqIndex); % ������ת���ɾ��������,ע�� x,y������

    % ��Ȼfscores���ֵ�������ظ�����ô�����ٿ�  acc ׼ȷ�� ָ��
    accFscore=[];
    maxAcc=[];
    accSen=[];
    accSpec=[];
    accPrec=[];   
    accAuc=[];
    accLdec={};
    accLtest={};
    accTP={};
    accFP={};
    
    accBest_X=[];   % ��¼���ֵX������
    accBest_Y=[];   % ��¼���ֵY������
    for i=1:length(x)  % ����fscores���������±��ҳ����ж�Ӧ������ָ��
        
        
        accBest_X=[accBest_X,x(i)];
        accBest_Y=[accBest_Y,y(i)];
        
        accFscore=[accFscore,matFscore(x(i),y(i))];
        maxAcc=[maxAcc,matAcc(x(i),y(i))];        
        accSen=[accSen,matSen(x(i),y(i))];
        accSpec=[accSpec,matSpec(x(i),y(i))];
        accPrec=[accPrec,matPrec(x(i),y(i))];        
        accAuc=[accAuc,matAuc(x(i),y(i))];
        
        accLdec=[accLdec,matLdec{x(i),y(i)}];
        accLtest=[accLtest,matLtest{x(i),y(i)}];
        accTP=[accTP,matTP{x(i),y(i)}];
        accFP=[accFP,matFP{x(i),y(i)}];
    end
    
    [~,maxMapIndex]=max(accFscore(:)); % ������� acc .�������ظ���.������ظ��ľ�ѡ���һ�������ٽ�һ��������ָ����
    accFscore=accFscore(maxMapIndex);
    maxAcc=maxAcc(maxMapIndex);
    accSen=accSen(maxMapIndex);
    accSpec=accSpec(maxMapIndex);
    accPrec=accPrec(maxMapIndex);   
    accAuc=accAuc(maxMapIndex);
    
    accLdec=accLdec{maxMapIndex};
    accLtest=accLtest{maxMapIndex};
    accTP=accTP{maxMapIndex};
    accFP=accFP{maxMapIndex};
    
    accBest_X=accBest_X(maxMapIndex);
    accBest_Y=accBest_Y(maxMapIndex);

end









