% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
addpath('../MV_datasets');
clear;
close all;
clc;
tic;

dataset = load('C101_p1474.mat');   %initial_neighbor = 91,adaptive_neighbor = 9, gamma1 = 1  gamma2 = 1
%dataset = load('ORL_mtv.mat');       %initial_neighbor = 39,adaptive_neighbor = 6, gamma1 = 1  gamma2 = 1
%dataset = load('COIL_20_ZCQ.mat');  %initial_neighbor = 10,adaptive_neighbor = 9, gamma1 = 1  gamma2 = 1

data = dataset.X_train;
groundtruth = dataset.truth;

% 如果想要在ORL_mtv.mat数据集上运行，用下面的语句，注释掉上面的语句
% data = dataset.X;
% data{1} = data{1}'; data{2} = data{2}'; data{3} = data{3}';
% groundtruth = dataset.Y;
% data = data';

numOfImage = size(data{1},2);                       % 图片数量
numOfView = length(data);                           % 视图数量
numOfCluster = length(unique(groundtruth));         % 实际的分类数量 
S_init = zeros(numOfImage,numOfImage,numOfView);    % 各个视图的similarity matrix   

initial_neighbor = 91;  % 用k-NN算法构造各个视图的similarity matrix时，需要给每个视图的结点分配初始"邻居"
adaptive_neighbor = 9;  % 用于Algorithm 1迭代更新S

% 画论文的figure2的
ACC_result = zeros(5,5);
ACC_row = 1;
ACC_column = 1;
% for g1 = -3:1
% for g2 = -3:1

% g1、g2写成循环的形式是为了方便验证该算法的参数不敏感性。作者在论文提到g1、g2可以变化
for g1 = -3:1
    for g2= -3:1
        gamma1 = 10^g1;
        gamma2 = 10^g2;
        for view = 1:numOfView
            S_init(:,:,view) = constructS(data{view},initial_neighbor);     % 用k-NN算法构造每个视图的similarity matrix
        end
        
        [ACC,NMI,Purity,Precision,Recall,F_score,ARI,cluster_num,kmeans_value] = GSF(S_init,numOfCluster,groundtruth,gamma1,gamma2,adaptive_neighbor);
        fprintf('Cluster num:%d\n',numOfCluster);
        fprintf('ACC:%f\n',ACC);
        fprintf('nmi: %f\n', NMI);
        fprintf('purity: %f\n', Purity);
        fprintf('Precision: %f\n', Precision);
        fprintf('Recall: %f\n', Recall);
        fprintf('F-score: %f\n', F_score);
        fprintf('ARI: %f\n', ARI);
        fprintf('kmeans value:%f\n',kmeans_value);      % 数值太小直接显示为0了，在Cluster方法可以看kmeans_value

        ACC_result(ACC_row, ACC_column) = ACC;
        ACC_column = ACC_column + 1;
        if (ACC_column > 5)
            ACC_column = 1;
            ACC_row = ACC_row + 1;
        end
    end
end


% 画出figure2
ACC_result = ACC_result * 100;
figure('position',[100,100,500,500]);
bar3(ACC_result,0.75);
set(gca,'yticklabel',{'-3','-2','-1','0','1'},'Fontname','Times New Roman','FontSize',11);
set(gca,'xticklabel',{'-3','-2','-1','0','1'},'Fontname','Times New Roman','FontSize',11);
ylabel('log gamma1','Fontname','Times New Roman','FontSize',14);
xlabel('log gamma2','Fontname','Times New Roman','FontSize',14);
zlabel('ACC','Fontname','Times New Roman','FontSize',14);

toc;


