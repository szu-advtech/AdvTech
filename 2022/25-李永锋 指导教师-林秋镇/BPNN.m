function init_Pop = BPNN(his_POS,curr_POS,outputdecs)

%%his_NDS: nxd curr_NDS: nxd

% BP神经网络结构参数
h=size(his_POS,2);  %input nodes 设置为个体决策变量维度
i=5;                %Hidden nodes   
j=size(curr_POS,2); %Output nodes 设置为个体决策变量维度
Alpha=0.01;          %The learning rate
Beta=0.01;           %The learning rate
Gamma=0.8;          %The constant determines effect of past weight changes
maxIteration=1;    %The max number of Iteration
trainNum=size(his_POS,1);


V=2*(rand(h,i)-0.5);    %The weights between input and hidden layers——[-1, +1]
W=2*(rand(i,j)-0.5);    %The weights between hidden and output layers——[-1, +1]
HNT=2*(rand(1,i)-0.5);  %The thresholds of hidden layer nodes
ONT=2*(rand(1,j)-0.5);  %The thresholds of output layer nodes
DeltaWOld(i,j)=0; %The amout of change for the weights  W
DeltaVOld(h,i)=0; %The amout of change for the weights  V
DeltaHNTOld(i)=0; %The amount of change for the thresholds HNT
DeltaONTOld(j)=0; %The amount of change for the thresholds ONT
% BP神经网络的构建过程
Epoch=1;


% Normalize the data set(对输入和输出进行归一化)

[inputn,inputs] = mapminmax(his_POS');
inputn = inputn';
[outputn,outputs] = mapminmax(curr_POS');
outputn = outputn';

while Epoch<maxIteration
    for k=1:trainNum
        %获取输入
        a=inputn(k,:);
        %设定期望输出
        ck=outputn(k,:);
        % Calcluate the value of activity of hidden layer FB
        for ki=1:i
            b(ki)=tansig(a*V(:,ki)+HNT(ki));
        end;
        %  Calcluate the value of activity of hidden layer FC
        for kj=1:j
            c(kj)=tansig(b*W(:,kj)+ONT(kj));
        end;
        % Calculate the errorRate of FC
%         d=c.*(1-c).*(ck-c);
        d=(1-c.*c).*(ck-c);
        
        % Calculate the errorRate of FB
%         e=b.*(1-b).*(d*W');
        e=(1-b.*b).*(d*W');
        %Update the weights between FC and FB——Wij 
        for ki=1:i
            for kj=1:j
                DeltaW(ki,kj)=Alpha*b(ki)*d(kj)+Gamma*DeltaWOld(ki,kj);
            end
        end;
        W=W+DeltaW;
        DeltaWOld=DeltaW;
        %Update the weights between FA and FB——Vhj
        for kh=1:h
            for ki=1:i
                DeltaV(kh,ki)=Beta*a(kh)*e(ki);                               
            end
        end;
        V=V+DeltaV;                                                    
        DeltaVold=DeltaV;                                              
        % Update HNT and ONT
        DeltaHNT=Beta*e+Gamma*DeltaHNTOld;
        HNT=HNT+DeltaHNT;
        DeltaHNTOld=DeltaHNT;
        DeltaONT=Alpha*d+Gamma*DeltaONTOld;
        ONT=ONT+DeltaONT;
        DeltaTauold=DeltaONT;
    end 
    Epoch = Epoch +1;
    if(d<=0.01)
        break;
    end% update the iterate number
end



inputn = outputdecs;

for k=1:size(inputn,1)
    a=inputn(k,:); %get testSet
    
    %Calculate the value of activity of hidden layer FB
    for ki=1:i
        b(ki)=logsig(a*V(:,ki)+HNT(ki));
    end;
    %Calculate the result
    for kj=1:j
        c(kj)=logsig(b*W(:,kj)+ONT(kj));
    end;
    
    init_Pop(k,:)=c;

end
init_Pop = mapminmax('reverse',init_Pop',inputs);
init_Pop = init_Pop';

end
