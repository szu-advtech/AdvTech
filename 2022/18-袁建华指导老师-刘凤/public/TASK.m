classdef TASK    
    %����Ϊ�������������Ϣ�������������������ά�ȣ�ͳһ�����ռ䣬�������½����������������Ҫ��initTASK��ʼ��
    properties
        M;%�������
        Tdims;%����ά��
        D_multitask;%ͳһ�����ռ�
        Lb;%������½�
        Ub;%������Ͻ�
        fun;%����
    end    
    methods        
        function object = initTASK(object,name)
            switch name
                case 'RastriginAckley'
                    object.M = 2;%��ʼ���������
                    object.Tdims = zeros(object.M,1);%��ʼ������ά��
                    object.Tdims(1) = 40;%Rastriginά��
                    object.Tdims(2) = 30;%Ackleyά��
                    object.D_multitask = max(object.Tdims);%ͳһ�����ռ�
                    object.Lb = ones(object.M,object.D_multitask);%��ʼ���½�
                    object.Ub = ones(object.M,object.D_multitask);%��ʼ���Ͻ�
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin�½�
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley�½�
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin�Ͻ�
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley�Ͻ�
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));%���������ı�׼������,rotation matrices
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));%���������ı�׼������,rotation matrices
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                case 'SphereWeierstrass'
                    object.M = 2;%��ʼ���������
                    object.Tdims = zeros(object.M,1);%��ʼ������ά��
                    object.Tdims(1) = 30;%Sphereά��
                    object.Tdims(2) = 30;%Weierstrassά��
                    object.D_multitask = max(object.Tdims);%ͳһ�����ռ�
                    object.Lb = ones(object.M,object.D_multitask);%��ʼ���½�
                    object.Ub = ones(object.M,object.D_multitask);%��ʼ���Ͻ�
                    object.Lb(1,:) = -100*object.Lb(1,:);%Sphere�½�
                    object.Lb(2,:) = -0.5*object.Lb(2,:);%Weierstrass�½�
                    object.Ub(1,:) = 100*object.Ub(1,:);%Sphere�Ͻ�
                    object.Ub(2,:) = 0.5*object.Ub(2,:);%Weierstrass�Ͻ�
                    MS=eye(object.Tdims(1),object.Tdims(1));
                    object.fun(1).fnc=@(x)Sphere(x,MS);
                    MW=orth(randn(object.Tdims(2),object.Tdims(2)));
                    object.fun(2).fnc=@(x)Weierstrass(x,MW);
                case 'RastriginAckleySphere'
                    object.M = 3;%��ʼ���������
                    object.Tdims = zeros(object.M,1);%��ʼ������ά��
                    object.Tdims(1) = 40;%Rastriginά��
                    object.Tdims(2) = 50;%Ackleyά��
                    object.Tdims(3) = 20;%Sphereά��
                    object.D_multitask = max(object.Tdims);%ͳһ�����ռ�
                    object.Lb = ones(object.M,object.D_multitask);%��ʼ���½�
                    object.Ub = ones(object.M,object.D_multitask);%��ʼ���Ͻ�
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin�½�
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley�½�
                    object.Lb(3,:) = -100*object.Lb(3,:);%Sphere�½�
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin�Ͻ�
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley�Ͻ�
                    object.Ub(3,:) = 100*object.Ub(3,:);%Sphere�Ͻ�
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                    MS=eye(object.Tdims(3),object.Tdims(3));
                    object.fun(3).fnc=@(x)Sphere(x,MS);
                case 'RastriginRastrigin'
                    object.M = 2;%��ʼ���������
                    object.Tdims = zeros(object.M,1);%��ʼ������ά��
                    object.Tdims(1) = 30;%Rastriginά��
                    object.Tdims(2) = 30;%Rastriginά��
                    object.D_multitask = max(object.Tdims);%ͳһ�����ռ�
                    object.Lb = ones(object.M,object.D_multitask);%��ʼ���½�
                    object.Ub = ones(object.M,object.D_multitask);%��ʼ���Ͻ�
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin�½�
                    object.Lb(2,:) = -5*object.Lb(2,:);%Rastrigin�½�
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin�Ͻ�
                    object.Ub(2,:) = 5*object.Ub(2,:);%Rastrigin�Ͻ�
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));%���������ı�׼������,rotation matrices
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    object.fun(2).fnc=@(x)Rastrigin(x,MR);
                case 'AckleyAckley'
                    object.M = 2;%��ʼ���������
                    object.Tdims = zeros(object.M,1);%��ʼ������ά��
                    object.Tdims(1) = 30;%Ackleyά��
                    object.Tdims(2) = 30;%Ackleyά��
                    object.D_multitask = max(object.Tdims);%ͳһ�����ռ�
                    object.Lb = ones(object.M,object.D_multitask);%��ʼ���½�
                    object.Ub = ones(object.M,object.D_multitask);%��ʼ���Ͻ�
                    object.Lb(1,:) = -32*object.Lb(1,:);%Ackley�½�
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley�½�
                    object.Ub(1,:) = 32*object.Ub(1,:);%Ackley�Ͻ�
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley�Ͻ�
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));%���������ı�׼������,rotation matrices
                    object.fun(1).fnc=@(x)Ackley(x,MA);
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                case 'RastriginAckleyLinear'
                    object.M = 3;%��ʼ���������
                    object.Tdims = zeros(object.M,1);%��ʼ������ά��
                    object.Tdims(1) = 40;%Rastriginά��
                    object.Tdims(2) = 50;%Ackleyά��
                    object.Tdims(3) = 30;%Linearά��
                    object.D_multitask = max(object.Tdims);%ͳһ�����ռ�
                    object.Lb = ones(object.M,object.D_multitask);%��ʼ���½�
                    object.Ub = ones(object.M,object.D_multitask);%��ʼ���Ͻ�
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin�½�
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley�½�
                    object.Lb(3,:) = -10*object.Lb(3,:);%Linear�½�
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin�Ͻ�
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley�Ͻ�
                    object.Ub(3,:) = 10*object.Ub(3,:);%Linear�Ͻ�
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                    MS=eye(object.Tdims(3),object.Tdims(3));
                    object.fun(3).fnc=@(x)Sphere(x,MS);
            end
        end  
    end
end