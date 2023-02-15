

% ��ȡ���е�.mat�ļ�

clear
resPath=cd;
allRes=dir([resPath,'\*.mat']);


len=length(allRes);


% select='Res1';      % ѡ�����Fscore
select='Res2';    % ѡ�����Acc

for i=1:len
    strName{i}=allRes(i).name;
    load (strName{i}); % ���ζ�ȡmat

    Res=eval(select); % ѡ�����Acc
    
    a(i,1)=Res.svmAcc;
    a(i,2)=Res.svmSen;
    a(i,3)=Res.svmSpec;
    a(i,4)=Res.svmPrec;
    a(i,5)=Res.svmFscore;
    a(i,6)=Res.svmAuc;
    a(i,7)=Res.maxAcc;
    a(i,8)=Res.minArmse;
    a(i,9)=Res.maxBcc;
    a(i,10)=Res.minBrmse;
end


a=real(a);
matName=strName';
titleName={'Feature','ACC','SEN','SPEC','PREC','Fscore','AUC','scoreAcc','ARMSE','scoreBcc','BRMSE'};

combine=[matName,num2cell(a)];
combine=[titleName;combine];

% xlsFile �����xls�ļ�·����

% if(exist(xlsFile,'file')) % ����Ѿ����ڣ���ô����ɾ��.
%    delete(xlsFile);
% end
xlsFile=['c:\',select,'.xlsx'];
xlswrite(xlsFile, combine); % ��need_xml_infoֱ��ת����xls��ʽ




