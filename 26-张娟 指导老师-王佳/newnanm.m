function net =newnanm(input_train,target_train,ita,M,epoch)
x=input_train;%4,45
t=target_train;%1,45

[I,J]=size(x);%I->features ������Ŀ4,  J->the size of samples������С 45
%fprintf(I)
w=2*(rand(I,M)-1/2);%M ��Ԫ��
q=2*(rand(I,M)-1/2);
qs=0.5;
k=5;
Y=zeros(I,M,J);
Z=ones(M,J);
V=zeros(1,J);
O=zeros(1,J);
E=zeros(J,epoch);%45,1000
dw=zeros(I,M,J);
dq=zeros(I,M,J);
dYw=zeros(I,M,J);
dYq=zeros(I,M,J);
dZY=zeros(I,M,J);
dVZ=zeros(M,J);
dOV=zeros(1,J);

for e=1:epoch
    for j=1:J %�������
        % build a connection layer  һ������
        for m=1:M
            for i=1:I
                Y(i,m,j)=1/(1+exp(-k*(w(i,m)*x(i,j)-q(i,m))));
            end
        end
        % build a dendritic layer
        for m=1:M
            Q=1;
            for i=1:I
                Q=Q*Y(i,m,j);
            end
            Z(m,j)=Q;
        end
        % build a menbrane layer
        V=sum(Z);
        % build a soma body
        O(j)=1/(1+exp(-k*(V(j)-qs)));%Ԥ��ֵ
        E(j,e)=1/2*((O(j)-t(j))^2); %Ŀ�꺯�� һ����������һ����ʧ
    end
    %-----��������ѵ��һ�Σ�����һ�η��򴫲�------
    %���򴫲�
    for j=1:J
        for m=1:M
            for i=1:I
                dYw(i,m,j)=k*x(i,j)*exp(-k*(w(i,m)*x(i,j)-q(i,m)))/((1+exp(-k*(w(i,m)*x(i,j)-q(i,m))))^2);
                dYq(i,m,j)=-k*exp(-k*(w(i,m)*x(i,j)-q(i,m)))/((1+exp(-k*(w(i,m)*x(i,j)-q(i,m))))^2);
                Q=1;
                for L=1:I
                    if(L~=i)
                        Q=Q*Y(L,m,j);
                    end
                    dZY(i,m,j)=Q;
                end
                dVZ(m,j)=1;
                dOV(j)=k*exp(-k*(V(j)-qs))/((1+exp(-k*(V(j)-qs)))^2);
                dEO=O(j)-t(j);
                dw(i,m,j)=dEO*dOV(j)*dVZ(m,j)*dZY(i,m,j)*dYw(i,m,j);
                dq(i,m,j)=dEO*dOV(j)*dVZ(m,j)*dZY(i,m,j)*dYq(i,m,j);
                w(i,m)=w(i,m)-ita*dw(i,m,j);
                q(i,m)=q(i,m)-ita*dq(i,m,j);
            end
        end
    end
%     disp(['The current mean error is :',num2str(mean(E(:,e))),'.'])%���ŵ�һά�����ֵ ѵ��һ�ֵ���ʧ
   
end
Error=mean(E);
plot(Error);
legend("Mean square Error");
xlabel("����");
ylabel("��ʧ");

net.w=w;
net.q=q;
net.Error=Error;
net.O=O;
end

