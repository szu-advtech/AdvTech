function [Accuracy, Threshold_logic] = evaluate_accuracy_logic_circuit( input, target, net, M, PS )
% Initialize the weights and threshold values
G = 1;
[I,J] = size(input);%2,4
Accuracy=zeros(1,G);
Threshold_logic=zeros(I,M,G);

for g=1:G
    w = net.w;
    q = net.q;
    
    K=zeros(I,M);
    Y=zeros(I,M,J);
    Z=ones(M,J);
    Output=zeros(1,J);
    
    Threshold=q./w;%突触层阈值  w q都是训练好的参数
    
    for m=1:M
        for i=1:I
            if (0<w(i,m)&&w(i,m)<q(i,m))
                K(i,m)=0;
            end
            if (w(i,m)<0&&q(i,m)>0)
                K(i,m)=0;
            end
            if (q(i,m)<0&&w(i,m)>0)
                K(i,m)=2;
            end
            if (q(i,m)<w(i,m)&&w(i,m)<0)
                K(i,m)=2;
            end
            if (w(i,m)<q(i,m)&&q(i,m)<0)
                K(i,m)=-1;
            end
            if (0<q(i,m)&&q(i,m)<w(i,m))
                K(i,m)=1;
            end
        end
    end
    
    %% Calculate the classification rate of the logic circuit 计算逻辑电路的分类率
    constant3=0;
    for j=1:J
        % build synaptic layers 根据w q值，确定连接情况，直接得出突触层输出的取值
        for m=1:M
            for i=1:I
                
                % Constant 1 connection
                if  K(i,m)==2
                    Y(i,m,j)=1;
                end
                
                % Constant 0 connection
                if  K(i,m)==0
                    Y(i,m,j)=0;
                end
                
                % Inverse connection
                if K(i,m)==-1
                    if input(i,j)<Threshold(i,m)%跟突触层的阈值比较
                        Y(i,m,j)=1;
                    else
                        Y(i,m,j)=0;
                    end
                end
                
                % Direct connection
                if K(i,m)==1
                    if input(i,j)<Threshold(i,m)
                        Y(i,m,j)=0;
                    else
                        Y(i,m,j)=1;
                    end
                end
            end
        end
        
        % build dendrite layers
        for m=1:M
            constant1=1;
            for i=1:I
                constant1=constant1&Y(i,m,j);%and 运算
            end
            Z(m,j)=constant1;
        end
        
        % build  membrane layers
        constant2=0;
        for m=1:M
            constant2=constant2|Z(m,j);%or运算
        end
        Output(j)=constant2;
        
        if Output(j)==target(j)
            constant3=constant3+1;
        end
    end
    
    Accuracy(g) =  constant3/J;%计算正确率
    Threshold_logic(:,:,g) = mapminmax('reverse',Threshold, PS);%threadshold 突触层阈值 反归一化
    
end
end


