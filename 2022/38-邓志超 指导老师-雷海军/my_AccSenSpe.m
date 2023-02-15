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

Not_nan_ind = ~isnan(yRec); % ������ֵ������ֵ�ų�
yRec = yRec(Not_nan_ind);
ylabel = ylabel(Not_nan_ind);
b = b(Not_nan_ind);

yp = sign(yRec+b); % >0��1 =0��0  <0��-1 
accuracy = sum(yp  == ylabel)./size(ylabel,1); % ׼ȷ�ȣ����Ϊ��ȷ����Ŀ(�����������͸�����)/����

true_pos = sum((yp == 1) .* (ylabel==1)); % ��⵽����������Ŀ,score>0 �ұ�ǩȷʵΪ����
sensitivity = true_pos/sum(ylabel==1); % R(�ٻ���)����������������/������lable����(������û�м�⵽)
sensitivity(find(isinf(sensitivity))) = eps; % ��������û������������ô����֡�
sensitivity(find(isnan(sensitivity))) = eps; 

true_neg = sum((yp == -1) .* (ylabel==-1)); % ��⵽�ĸ�������Ŀ,score<0 �ұ�ǩȷʵΪ����
specificity = true_neg/sum(ylabel==-1);
specificity(find(isinf(specificity))) = eps;  % ��������û�и���������ô����֡�
specificity(find(isnan(specificity))) = eps;

% calculate the F-score
Precision = true_pos/sum(yp == 1); %  ���ȣ�������ȷ��������/��������������
Precision(find(isinf(Precision))) = eps; % ��������û������������ô����֡�
Precision(find(isnan(Precision))) = eps;
Recall = sensitivity; % R(�ٻ���)
n = 1^2; % weight between precision and recall for the F-score
Fscore = (1+n)*Precision*Recall/(n*Precision+Recall); % f-measure=2PR/(P+R)
Fscore(find(isinf(Fscore))) = eps;  % ������������ͷ�����
Fscore(find(isnan(Fscore))) = eps;
