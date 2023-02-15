clear;
clear;

place='houston';
class_num=15;

% place='dioni';
% class_num=12;

% place='pavia';
% class_num=9;

model_list=["encoder32";"encoder64";"encoder64_encoder32";"nsct32";"nsct64";"nsct64_nsct32";"spectral_encoder64_encoder32";"spectral_nsct64_nsct32";"spectral"];

% for kk=1:9
    % new_path=strcat('SPregular-',place,'-',model_list(kk,:),'-RF-zyx');
    % old_path=strcat(place,'-',model_list(kk,:),'-RF-');
    new_path=strcat('SPregular-',place,'-','spectral_encoder64_encoder32_nsct64_nsct32','-RF-zyx');
    old_path = strcat('TEST',place,'-','spectral_encoder64_encoder32_nsct64_nsct32','-RF-');
    load(strcat('sample_table_',place,'_a_7.mat'),'st222');
    sp_num=max(st222(:,5))+1;
    if ~exist(new_path,'dir')
       mkdir(new_path);
    end
    for ii=5:18
        for pp=1:5
            % file_name=strcat(old_path,'/',place,'_',model_list(kk,:),'_train_label_ii',num2str(ii),'_pp',num2str(pp),'_best_model_predictY.mat');
            file_name=strcat(old_path,'/',place,'_','spectral_encoder64_encoder32_nsct64_nsct32','_train_label_ii',num2str(ii),'_pp',num2str(pp),'_best_model_predictY.mat');
            load(file_name,'predict_y');
            sp_px_stat=zeros(sp_num,class_num);
            for mm=1:size(st222,1)
                sp_px_stat(st222(mm,5)+1,predict_y(mm,1)+1)=sp_px_stat(st222(mm,5)+1,predict_y(mm,1)+1)+1;
            end
            [max_num,index_max]=max(sp_px_stat,[],2);
            clear predict_y;
            predict_y=zeros(size(st222,1),1);
            for mm=1:size(st222,1)
                predict_y(mm,1)=index_max(st222(mm,5)+1,1)-1;
            end
            save(strcat(new_path,'/',place,'_train_label_ii',num2str(ii),'_pp',num2str(pp),'_RF_predictY.mat'),'predict_y');
        end
    end
% end

