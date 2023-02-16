function CR=CR_calculation(obtained_ps,reference_ps)
% CR_calculation:计算得到的帕累托集的覆盖率

n_var=size(reference_ps,2);
% 在自己得到的帕累托最优解集的每个维度上求最大值和最小值
obtained_min=min(obtained_ps);  %[1.019	-0.99]，自己算出的ps（第一等级的所有个体）中x1和x2上分别的最小值
obtained_max=max(obtained_ps);  %[2.99	0.99]
% 在参考/真实帕累托集
reference_min=min(reference_ps);  %[1  -0.99]，真实帕累托最优解集中x1和x2上分别的最小值
reference_max=max(reference_ps);  %[3  0.99]
for i=1:n_var
    if reference_max(i)==reference_min(i) % 如果 reference_max的第一个变量 等于 reference_min的第一个变量
        kesi(i)=1; % 如果真实和得到的第i变量都相同，则忽略第i维。
    elseif obtained_min(i)>=reference_max(i)||reference_min(i)>=obtained_max(i) % 如果最小值大于最大值
         kesi(i)=0;
    else
        kesi(i)=((min(obtained_max(i),reference_max(i))-max(obtained_min(i),reference_min(i)))/...
        (reference_max(i)-reference_min(i)))^2;  %（obtained_max和reference_max的第一个变量的最小值）-（obtained_min和reference_min的第一个变量的最大值）/...，结果放在kesi的第一行第一列，i=2，结果放在kesi(1*2)的第一行第2列
    end
end
CR=nthroot(prod(kesi),2*n_var); % kesi是[0.9860，0.9963]，prod(kesi)是两元素相乘 即0.9823，nthroot(X,n)返回X元素的第n次实根，即0.9823的4次方
