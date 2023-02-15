clc;
clear;
close;

%% Load dataset
dataName = 'bibtex.mat';
load(dataName)
disp(data.infm);

%% A demo for MUSER algorithm
% ref: Partial Multi-Label Learning via Multi-Subspace Representation(ijcai-2020)
% STEP 1: Set the parameters
pars.Alpha = 1e-2;
pars.Beta = 1e-2;
pars.Gamma = 1e0;
pars.c = floor(0.5*data.infm.nr_class);
pars.m = floor(0.5*data.infm.dim);
pars.lr_rate = 0.2;
pars.T_max = 50;
pars.algo = 0;  %default:0;
% pars.algo = 1 represents the algorithms of mMUSER, a modification version

options = [];
options.NeighborMode = 'KNN';
options.k = 30;
options.WeightMode = 'HeatKernel';
options.t = 1;
W = constructW(data.data,options);
pars.L = W;

%STEP 2: Produce the redundant labels for each instance
pars.Y = data.label;  
for i = 1:size(data.label,1)
    ind = find(data.label(i,:)==0);
    tmp = randperm(length(ind),2);   %r = 1, 2, or 3
    pars.Y(i,ind(tmp)) = 1;
end

[result] = MUSER_train(data, pars);

[pre] = MUSER_predict(data,result);
[criteria] = CriteriaForMLL(pre, data);
disp(criteria)
%% This is an example for searching best parameters using grid method
% tic
% data.data = normalize(data.data,'range');
% result_table = zeros(3*3*3*3,9);
% q = 0;
% for i1 = -2:2:2  %Alpha
%     for i2 = -2:2:2  %Beta
%         for i3 = -2:2:2%Gamma
%             for k = 10:20:50
%                 try
%                     options = [];
%                     options.NeighborMode = 'KNN';
%                     options.k = k;
%                     options.WeightMode = 'HeatKernel';
%                     options.t = 1;
%                     W = constructW(data.data,options);
%                     pars.L = W;
%                     pars.Alpha = 10^i1;
%                     pars.Beta = 10^i2;
%                     pars.Gamma = 10^i3;
%                     [result] = MUSER_train(data, pars);
%                     [pre] = MUSER_predict(data,result);
%                     [criteria] = CriteriaForMLL(pre, data);
%                     q = q+1;
%                     result_table(q,:) = [10^i1,10^i2,10^i3,k,criteria.hammloss,...
%                         criteria.rankloss, criteria.one_er, criteria.Coverage_ac,...
%                         criteria.avg_prec];
%                     disp(['Times:', num2str(q)])
%                     disp(criteria)
%                 catch
%                     disp(['Times:', num2str(q)])
%                     disp('wrong in somewhere!')
%                 end
%             end
%         end
%     end
% end
% toc
%%
% tmp = result_table(:,5:8);
% tmp(tmp==0) = 1e7;
% min(tmp)  % The best result for: hammloss, rankloss, one_er and Coverage_ac
% tmp = result_table(:,9);  
% tmp(tmp==0) = 0; 
% max(tmp) %The best result for: avg_prec

%% TSNE visualization
rng default
Q = result.Q;
W = result.W;
% You have TWO choices: X*Q or X*Q*W
Y_MUSER = data.data*Q;
% Y_MUSER = data.data*Q*W;

Y = tsne(Y_MUSER);
Y_tsne = [];
Y_label_tsne = [];
for i = 1:data.infm.nr_class
    if sum(data.label(:,i) == 1) == 0
        continue
    end
    try
    Y_tsne = [Y_tsne; Y(data.label(:,i) == 1,:)];
    Y_label_tsne = [Y_label_tsne; ones(sum(data.label(:,i) == 1),1)*i];
    catch
        disp(i)
    end
end
gscatter(Y_tsne(:,1),Y_tsne(:,2),Y_label_tsne)
legend('off')
%% Function values figure
figure(2)
plot(result.funcvalue)
xlabel('Iterations')
ylabel('Function value')