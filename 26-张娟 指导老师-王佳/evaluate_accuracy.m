function [accuracy,Synapse_number] = evaluate_accuracy( x, T, pop, M )

[II,J] = size(x);%2,4   4,45
[G,~] = size(pop);%ģ��
k = 5;
O = zeros(J,G);
accuracy = zeros(1,G);
Synapse_number = zeros(1,G);
for g=1:G
    
    w=zeros(II,M);                  % the weight value of dendrites
    q=zeros(II,M);                  % the threshold value of dendrites
    w(:,:)=pop(g).w;%��ֵ ����ģ�͵Ĳ���
    q(:,:)=pop(g).q;
    Y=zeros(II,M,J);
    Z=ones(M,J);
    V=zeros(1,J);
    constant1=0;
    
    %% %%%%%%%%%%%%%  Calculation Fitness one %%%%%%%%%%%%%
    for j=1:J
        % build synaptic layers
        for m=1:M
            for i=1:II
                Y(i,m,j)=1/(1+exp(-k*(w(i,m)*x(i,j)-q(i,m))));
            end
        end
        
        % build dendrite layers
        for m=1:M
            Q=1;
            for i=1:II
                Q=Q*Y(i,m,j);
            end
            Z(m,j)=Q;
        end
        
        % build  membrane layers �ۼ�
        constant=0;
        for m=1:M
            constant=constant+Z(m,j);
        end
        V(j)=constant;
        
        % build a soma layer ϸ����
        O(j,g)=1/(1+exp(-k*(V(j)-0.5)));
        
        if O(j,g)>0.5
            O(j,g)=1;
        else
            O(j,g)=0;
        end
        
        if O(j,g)==T(j)
           constant1=constant1+1;%Ԥ��ֵ=��ʵֵ����1
        end
    end

    accuracy(g) =  constant1/J;%������ȷ��

    %% %%%%%%%%%%%%%  Calculation Fitness two %%%%%%%%%%%%%
    for m=1:M
        for i=1:II
            if (0<w(i,m)&&w(i,m)<q(i,m))%K ��¼�������:��0���� ��1���� ��״��4,8��  ����w q ��ֵ������������
                K(i,m,g)=0;    % constant 0
            end
            if (w(i,m)<0&&q(i,m)>0)
                K(i,m,g)=0;    % constant 0
            end
            if (q(i,m)<0&&w(i,m)>0)
                K(i,m,g)=2;    % constant 1
            end
            if (q(i,m)<w(i,m)&&w(i,m)<0)
                K(i,m,g)=2;    % constant 1
            end
            if (w(i,m)<q(i,m)&&q(i,m)<0)
                K(i,m,g)=-1;   % Direct
            end
            if (0<q(i,m)&&q(i,m)<w(i,m))
                K(i,m,g)=1;    % Inverse
            end
        end
    end
    
    Left_synapse=II*M;%ͻ����������������24
    for m=1:M
        canstant=1;  sestant=0;
        for i=1:II
            canstant=canstant*K(i,m,g);
            if K(i,m,g)==2
                sestant= sestant+1;%yһ�����¼��1 ���ӵĸ���
            end
        end
        if canstant~=0 %��һ���ﲻ����constane 0����ʱ
            synapse = sestant;
        else
            synapse = II;
        end
        Left_synapse=Left_synapse-synapse;%��ͻ�޼�  ��1 ���ӿ��Ժ���
     end

    Synapse_number(g)=Left_synapse; %4  4
end

