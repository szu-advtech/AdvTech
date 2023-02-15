function [fsAcc,fsSen,fsSpec,fsPrec,fsAuc,maxFscore,fsLdec,fsLtest,fsTP,fsFP,fsBest_X,fsBest_Y,maxAuc,aucLdec,aucLtest,aucTP,aucFP,aucBest_X,aucBest_Y]=...
    my_findBestFscore_SVM(matAcc,matSen,matSpec,matPrec,matAuc,matFscore,matLdec,matLtest,matTP,matFP)


    % -------------------- AUC -----------------------------
    [maxAuc,maxAucIndex]=max(matAuc(:)); % �ҳ����fscores.�������ظ���
    [aucBest_X,aucBest_Y]=ind2sub(size(matAuc),maxAucIndex); % ������ת���ɾ��������,ע�� x,y������
    aucLdec=matLdec{aucBest_X,aucBest_Y};
    aucLtest=matLtest{aucBest_X,aucBest_Y};
    aucTP=matTP{aucBest_X,aucBest_Y};
    aucFP=matFP{aucBest_X,aucBest_Y};
    
    
    % -------------------- Fscore -----------------------------
    maxFscore=max(matFscore(:)); % �ҳ����fscores.�������ظ���
    seqIndex=find(matFscore==maxFscore); % �ҳ������ظ������ֵ��˳������(����)
    [x,y]=ind2sub(size(matFscore),seqIndex); % ������ת���ɾ��������,ע�� x,y������

    % ��Ȼfscores���ֵ�������ظ�����ô�����ٿ�  acc ׼ȷ�� ָ��
    maxFscore=[];
    fsAcc=[];
    fsSen=[];
    fsSpec=[];
    fsPrec=[];   
    fsAuc=[];
    fsLdec={};
    fsLtest={};
    fsTP={};
    fsFP={};
    
    fsBest_X=[];   % ��¼���ֵX������
    fsBest_Y=[];   % ��¼���ֵY������
    for i=1:length(x)  % ����fscores���������±��ҳ����ж�Ӧ������ָ��
        
        
        fsBest_X=[fsBest_X,x(i)];
        fsBest_Y=[fsBest_Y,y(i)];
        
        maxFscore=[maxFscore,matFscore(x(i),y(i))];
        fsAcc=[fsAcc,matAcc(x(i),y(i))];        
        fsSen=[fsSen,matSen(x(i),y(i))];
        fsSpec=[fsSpec,matSpec(x(i),y(i))];
        fsPrec=[fsPrec,matPrec(x(i),y(i))];        
        fsAuc=[fsAuc,matAuc(x(i),y(i))];
        
        fsLdec=[fsLdec,matLdec{x(i),y(i)}];
        fsLtest=[fsLtest,matLtest{x(i),y(i)}];
        fsTP=[fsTP,matTP{x(i),y(i)}];
        fsFP=[fsFP,matFP{x(i),y(i)}];
    end
    
    [~,maxMapIndex]=max(fsAcc(:)); % ������� acc .�������ظ���.������ظ��ľ�ѡ���һ�������ٽ�һ��������ָ����
    maxFscore=maxFscore(maxMapIndex);
    fsAcc=fsAcc(maxMapIndex);
    fsSen=fsSen(maxMapIndex);
    fsSpec=fsSpec(maxMapIndex);
    fsPrec=fsPrec(maxMapIndex);   
    fsAuc=fsAuc(maxMapIndex);
    
    fsLdec=fsLdec{maxMapIndex};
    fsLtest=fsLtest{maxMapIndex};
    fsTP=fsTP{maxMapIndex};
    fsFP=fsFP{maxMapIndex};
    
    fsBest_X=fsBest_X(maxMapIndex);
    fsBest_Y=fsBest_Y(maxMapIndex);

end









