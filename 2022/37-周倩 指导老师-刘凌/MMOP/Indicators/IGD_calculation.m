function  IGD=IGD_calculation(obtained_ps,reference_ps)
% IGD:计算得到的Pareto集的IGD
% reference_ps PS ：真实PS

n_ref=size(reference_ps,1);  % 400

for i=1:n_ref  % i=1:400
    ref_m=repmat(reference_ps(i,:),size(obtained_ps,1),1);  % size(obtained_ps,1)是200，把reference_ps第一行的两个数（第一个个体的两个决策变量）往下复制200份，变成200*2。i=2，把reference_ps第2行（第2个个体）的两个数往下复制200份
    %i=1时，计算得到的所有个体与第1行（第一个个体）的距离。i=2，计算得到的所有个体与第2行（第2个个体）的距离。
    d=obtained_ps-ref_m;    
    D=sum(abs(d).^2,2).^0.5; %取d（两列）的绝对值 平方，两列相加，根号（200*1）
    [~,b(i)] = min(D);  % b(1)存放D中最小值的编号，i=2，b(2)存放D中最小值的编号（这些编号可能会一样，因为D是会变的）
    obtained_to_ref(i)=min(D);  % 第1行1列存放D的最小值（即与第1个个体的最小距离），i=2，第1行2列存放D的最小值（即与第2个个体的最小距离）。obtained_to_ref存放每次迭代后D中的最小值，1*400
end
IGD=sum(obtained_to_ref)/n_ref; % 求平均
end