%结果展示
function resultShow()
clear;
clc;

files=dir(fullfile('result','*.mat'));
fileNames={files.name};

for n = 1:numel(fileNames)
    dataName = fileNames{n};
    load(['result\' dataName]);
    i = find('.'==dataName);
    dataName=dataName(1:i-1);

    featureNum = zeros(foldNum,repeat);
    acc = zeros(foldNum,repeat);

    for run = 1:repeat
       res = result{run};
       for fold = 1:foldNum
          featureNum(fold,run) = res{fold}.featureNum ;
          acc(fold,run) = res{fold}.acc ;
       end
    end
    resFeatureNum = mean(featureNum);
    resMeanAcc = mean(acc);

    Best_Acc=max(resMeanAcc);

    disp([dataName '   repeat_times == ' num2str(repeat)]);%重复次数
    disp([dataName '   Best:  Accuracy == ' num2str(Best_Acc)]);%最佳精确度
    disp([dataName '   mean:  Accuracy == ' num2str(mean(resMeanAcc))]);%平均精确度
    disp([dataName '   mean:  featureSize == ' num2str(mean(resFeatureNum)) ]);%特征子集平均大小
    disp([dataName '   mean:  total_time:  == ' num2str(total_time/repeat/10)]);%平均训练时间
    disp([dataName '   std:   Accuracy == ' num2str(std(acc(:)))]);%精度标准差
    disp(' ');%空行

%     %下面这行方便导出，上面几行方便显示
%     disp([dataName,',',num2str(repeat),',',num2str(Best_Acc),',',num2str(mean(resMeanAcc)),',',num2str(mean(resFeatureNum)),',',num2str(total_time/repeat/10),',',num2str(std(acc(:)))]);%重复次数


end