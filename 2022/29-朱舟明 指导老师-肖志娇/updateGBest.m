%获取全局最佳
function [gBest,flag] = updateGBest(gBest,pos,fitness,subset,skillFactor)
 
    flag = false;
    i = skillFactor==1;
    fit1 = fitness;
    fit2 = fitness;
    fit1(~i) = 1;
    fit2(i) = 1;
    [~,index1] = min(fit1);
    [~,index2] = min(fit2);
    if gBest.task1.fit > fit1(index1)
        gBest.task1.pos = pos(index1,:); 
        gBest.task1.fit = fit1(index1);     
        gBest.task1.mask = subset;
        flag = true;
    end
    if gBest.task2.fit > fit2(index2)
       gBest.task2.pos = pos(index2,:); 
       gBest.task2.fit = fit2(index2);
    end
end

