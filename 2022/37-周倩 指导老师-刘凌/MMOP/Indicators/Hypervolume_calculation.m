function hyp=Hypervolume_calculation(pf,repoint)
% repoint是[1.1 1.1]，是MMF1测试集自带的

popsize=size(pf,1); % 200*2
[~,temp_index]=sort(pf(:,1));  %对第1列f1进行升序排序，temp_index是排序后对应的个体编号（200*1）
sorted_pf=pf(temp_index,:);  % sorted_pf是按第1列f1排序后的结果（200*2）
pointset=[repoint;sorted_pf];  %把这两个按行拼接起来（201*2）
hyp=0;

for i=1:popsize  %i=1:200
    q=(pointset(1,1)-pointset(i+1,1))*(pointset(i,2)-pointset(i+1,2)); %i=1时，(第1行1列减去第2行1列)*(第1行2列减去第2行2列)；i=2时，(第1行1列减去第3行1列)*(第2行2列减去第3行2列)
    hyp=hyp+q;  %把每次迭代得到的q加起来
end