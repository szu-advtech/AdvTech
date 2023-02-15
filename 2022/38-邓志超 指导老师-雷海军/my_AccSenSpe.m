function [accuracy,sensitivity,specificity,Fscore,Precision,true_pos,true_neg] = my_AccSenSpe(yRec,ylabel,varargin)

if isempty(varargin)
    b = zeros(length(yRec));
else 
    b = varargin{1};
end
% accuracy = sum((yRec>=0) == (ylabel==1))./size(ylabel,1);
% true_pos = sum((yRec>=0) .* (ylabel==1));
% sensitivity = true_pos/sum(ylabel==1);
% true_neg = sum((yRec<0) .* (ylabel==-1));
% specificity = true_neg/sum(ylabel==-1);

Not_nan_ind = ~isnan(yRec); % 保留数值，非数值排除
yRec = yRec(Not_nan_ind);
ylabel = ylabel(Not_nan_ind);
b = b(Not_nan_ind);

yp = sign(yRec+b); % >0置1 =0置0  <0置-1 
accuracy = sum(yp  == ylabel)./size(ylabel,1); % 准确度，检测为正确的数目(包括正样本和负样本)/总数

true_pos = sum((yp == 1) .* (ylabel==1)); % 检测到的正样本数目,score>0 且标签确实为正的
sensitivity = true_pos/sum(ylabel==1); % R(召唤率)：检索到的正样本/正样本lable个数(不管有没有检测到)
sensitivity(find(isinf(sensitivity))) = eps; % 若本来就没有正样本，那么会出现∞
sensitivity(find(isnan(sensitivity))) = eps; 

true_neg = sum((yp == -1) .* (ylabel==-1)); % 检测到的负样本数目,score<0 且标签确实为负的
specificity = true_neg/sum(ylabel==-1);
specificity(find(isinf(specificity))) = eps;  % 若本来就没有负样本，那么会出现∞
specificity(find(isnan(specificity))) = eps;

% calculate the F-score
Precision = true_pos/sum(yp == 1); %  精度：检索正确的正样本/检索到的正样本
Precision(find(isinf(Precision))) = eps; % 若本来就没有正样本，那么会出现∞
Precision(find(isnan(Precision))) = eps;
Recall = sensitivity; % R(召唤率)
n = 1^2; % weight between precision and recall for the F-score
Fscore = (1+n)*Precision*Recall/(n*Precision+Recall); % f-measure=2PR/(P+R)
Fscore(find(isinf(Fscore))) = eps;  % 避免出现无穷大和非数字
Fscore(find(isnan(Fscore))) = eps;
