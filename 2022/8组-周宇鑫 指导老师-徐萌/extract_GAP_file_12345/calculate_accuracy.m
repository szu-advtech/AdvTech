clear;
clear;

place='houston';
class_num=15;

% place='dioni';
% class_num=12;

% place='pavia';
% class_num=9;

model_list=["spectral";"encoder32";"encoder64";"nsct32";"nsct64";"encoder64_encoder32";"nsct64_nsct32";"spectral_encoder64_encoder32";"spectral_nsct64_nsct32"];
OAOAOA=[];
KCKCKC=[];

result_path = 'result';
if ~exist(result_path,'dir')
    mkdir(result_path);
end

% for kk=1:9
    confusion_cell=cell(1,1);
    producer_cell=cell(1,1);
    user_cell=cell(1,1);
    OA_cell=zeros(1,1);
    AA_cell=zeros(1,1);
    Kappa_cell=zeros(1,1);
    
    % y_path=strcat('SPregular-',place,'-',model_list(kk,:),'-RF-zsy2');
    y_path = strcat('SPregular-',place,'-','spectral_encoder64_encoder32_nsct64_nsct32','-RF-zyx');

    for ii=5:18
        for pp=1:5
            file_name=strcat(y_path,'/',place,'_train_label_ii',num2str(ii),'_pp',num2str(pp),'_RF_predictY.mat');
            load(file_name,'predict_y');
            load(strcat('sample_type_',place,'.mat'),'sample_type');
            predict_label=predict_y;
            test_label=sample_type;

            confusionmatrix=zeros(class_num,class_num);
            for k=1:size(test_label,1)
                confusionmatrix(predict_label(k,1)+1,test_label(k,1)+1)=confusionmatrix(predict_label(k,1)+1,test_label(k,1)+1)+1;
            end
            %计算 总体精度overall 
            %     用户判别精度user
            %     制图精度producer
            %     Kappa系数
            L=size(confusionmatrix,1);
            goundtruthmatrix=sum(confusionmatrix,1)';%真实地物参考图中每一类的样本数  列求和
            outcomematrix=sum(confusionmatrix,2);%分类图中每一类的样本数  行求和

            duijiaoxian=zeros(L,1);
            for i=1:L
               duijiaoxian(i,:)=confusionmatrix(i,i);
            end
            trueLabel = sum(confusionmatrix,1)';
            perAcc = duijiaoxian ./ trueLabel;
            aa = mean(perAcc);

            N=sum(goundtruthmatrix);
            overall=sum(duijiaoxian)/N;

            user=duijiaoxian./goundtruthmatrix;
            producer=duijiaoxian./outcomematrix;

            a=sum(goundtruthmatrix.*outcomematrix)/(N*N);
            kappa=(overall-a)/(1-a);

            OA_cell(ii,pp)=overall;
            Kappa_cell(ii,pp)=kappa;
            AA_cell(ii,pp)=aa;
            producer_cell{ii,pp}=producer;
            user_cell{ii,pp}=user;
            confusion_cell{ii,pp}=confusionmatrix;
        end
    end

    % save(strcat(file_name,'_confusion_cell.mat'),'confusion_cell');
    % save(strcat(file_name,'_producer_cell.mat'),'producer_cell');

    % save(strcat(place,'_',model_list(kk,:),'_user_cell.mat'),'user_cell');
    % save(strcat(place,'_',model_list(kk,:),'_OA_cell.mat'),'OA_cell');
    % OAOAOA(:,kk)=mean(OA_cell,2);
    % save(strcat(place,'_',model_list(kk,:),'_Kappa_cell.mat'),'Kappa_cell');
    % KCKCKC(:,kk)=mean(Kappa_cell,2);
    save(strcat(result_path,'/',place,'_','spectral_encoder64_encoder32_nsct64_nsct32','_user_cell.mat'),'user_cell');
    save(strcat(result_path,'/',place,'_','spectral_encoder64_encoder32_nsct64_nsct32','_OA_cell.mat'),'OA_cell');
    save(strcat(result_path,'/',place,'_','spectral_encoder64_encoder32_nsct64_nsct32','_Kappa_cell.mat'),'Kappa_cell');


    % save(strcat(file_name,'_AA_cell.mat'),'AA_cell');
% end





