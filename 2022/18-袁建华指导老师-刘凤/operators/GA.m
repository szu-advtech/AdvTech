function Offspring = GA(Parent,rmp,Pc,disC,sigma)
% �˺���������ͨ��ģ������ƽ���͸�˹��������Ӵ��������ô�ֱ�Ļ��������м������ӵļ̳У�������ɵĸ������壬������н�����߱��죩��
% Input: Parent������Ϣ��Ⱦɫ�壬�������ӣ���rmp�Ļ�����������Pcģ������ƽ�����ʡ�disC���������sigma��˹�������
% Output: Offspring�Ӵ���Ϣ��Ⱦɫ�壬�������ӣ�
% �ڡ�1.��ģʽ�У�ͨ���������������������ܾ�����ͬ�ĸ�ĸ���ڡ�2.��ģʽ�У�ͨ��������������������ĸһ����ͬ
    [N,~] = size(Parent.rnvec);
    select = randperm(N);
    rrnvec = Parent.rnvec(select,:);%����˳��
    sskill_factor = Parent.skill_factor(select,:); 
    Parent1 = rrnvec(1:floor(end/2),:);
    factor1 = sskill_factor(1:floor(end/2),:);
    Parent2 = rrnvec(floor(end/2)+1:floor(end/2)*2,:); % floor:����������
    factor2 = sskill_factor(floor(end/2)+1:floor(end/2)*2,:);
    Offspring = INDIVIDUAL();
    Offspring.skill_factor = sskill_factor;%��ʼ���Ӵ��ļ������Ӷ�ӦΪ�����ļ�������
    factorb1 = repmat(1:N/2,1,2);
    factorb2 = repmat(N/2+1:N,1,2);
    temp = randi(2,1,N);%�����Ӵ����ѡ�����Ǽ̳е�һ����ĸ���ǵڶ�����ĸ
    offactor = zeros(1,N);
    offactor(temp == 1) = factorb1(temp == 1);
    offactor(temp == 2) = factorb2(temp == 2);%�Ӵ��̳и�ĸ�ı��
    [NN,D]   = size(Parent1);
    
    % Simulated binary crossover
    beta = zeros(NN,D);
    mu   = rand(NN,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta(repmat(factor1 ~= factor2 & rand(NN,1)>=rmp,1,D)) = 1;%��ͬ�������ӵĸ���ֻ������rmp���ܽ���
    beta(repmat(rand(NN,1)>=Pc,1,D)) = 1;
    Offspring.rnvec = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
    Offspring.skill_factor(repmat(beta(:,1) ~= 1,2,1)) = sskill_factor(offactor(repmat(beta(:,1) ~= 1,2,1)));%���ڽ���ĸ��壬�����ѡ��һ�������̳л���
    
    % mutation
    rvec=normrnd(0,sigma,[N,D]);
    Offspring.rnvec(repmat(beta(:,1) == 1,2,D)) = Offspring.rnvec(repmat(beta(:,1) == 1,2,D)) + rvec(repmat(beta(:,1) == 1,2,D));%ֻ��û�н���ĸ�����б���
    %����
    Offspring.rnvec(Offspring.rnvec>1)=1;
    Offspring.rnvec(Offspring.rnvec<0)=0;
end