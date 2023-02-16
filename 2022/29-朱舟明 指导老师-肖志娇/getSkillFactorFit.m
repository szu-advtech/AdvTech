%获取技能因子和fit适应度，fitness是标量适应度（衡量样本对skillFactor任务的解决能力）
function [skillFactor,fitness,fit1,fit2] = getSkillFactorFit(dataX,dataY,pos,subset)

    alpha1 = 0.999999;
    alpha2 = 0.9;
    mask = repmat(subset,size(subset,1),1);
    
    %用五重交叉验证来评估训练集的准确度
    [featureNum1,error1] = knn5foldFast(dataX,dataY,(pos.*mask)>0.6);%(pos.*mask)>0.6是只取个体中subset中为1的列对应的位置，然后计算每个位置是否>0.6，传bool值过去
    [featureNum2,error2] = knn5foldFast(dataX,dataY,pos>0.6);%直接计算每个位置是否>0.6，传bool值
    
    fit1 = alpha1 * error1 + (1-alpha1)*(featureNum1./sum(mask));
    fit2 = alpha2 * error2 + (1-alpha2)*(featureNum2./size(dataX,2));

    [~,idx1] = sort(fit1,'des');
    [~,rank1] = sort(idx1);
    
    [~,idx2] = sort(fit2,'des');
    [~,rank2] = sort(idx2);
    [fitness,skillFactor] = min([rank1 rank2],[],2);

end
    