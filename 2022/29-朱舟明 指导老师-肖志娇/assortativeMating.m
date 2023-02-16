%随机交叉
function [V] = assortativeMating(V,pos,skillFactor,pbest,gbest,c1,c2,c3,w,rmp)

    select = (rand(size(V,1),1)<rmp);%小于rmp就用有c3的公式，就交叉
    index = select & skillFactor==1;%任务1，c3的时候用的task2的gbest
    V(index,:) = w*V(index,:) + c1*rand*( pbest.task1.pos(index,:)- pos(index,:)) + ...
        c2*rand*(gbest.task1.pos - pos(index,:)) + c3*rand*(gbest.task2.pos-pos(index,:));
    
    index = select & skillFactor==2;%任务2
    V(index,:) = w*V(index,:) + c1*rand*( pbest.task2.pos(index,:)- pos(index,:)) + ...
        c2*rand*(gbest.task2.pos - pos(index,:)) + c3*rand*(gbest.task1.pos-pos(index,:));

    index = ~select & skillFactor==1;
    V(index,:) = w*V(index,:) + c1*rand*( pbest.task1.pos(index,:)- pos(index,:)) + ...
        c2*rand*(gbest.task1.pos - pos(index,:));
    
    index = ~select & skillFactor==2;
    V(index,:) = w*V(index,:) + c1*rand*( pbest.task2.pos(index,:)- pos(index,:)) + ...
        c2*rand*(gbest.task2.pos - pos(index,:));
end

