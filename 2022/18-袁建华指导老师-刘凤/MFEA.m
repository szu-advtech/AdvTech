clc,clear
tic
%% ��������
global N gen
N = 50;                                     % ��Ⱥ��С
rmp = 0.3;                                  % �������ظ���
pi_l = 1;                                   % ����ѧϰ�ĸ���(BFGA quasi-Newton Algorithm)
Pc = 1;                                     % ģ������ƽ������
mu = 10;                                    % ģ������ƽ������
sigma = 0.02;                               % ��˹����ģ�͵ı�׼��
gen = 100;                                  %��������
selection_process = 'elitist';              % �ɹ�ѡ��elitist��roulette wheel��Tournament
name = 'RastriginAckleyLinear';             % ��������ѡ���У�RastriginAckley��SphereWeierstrass��RastriginAckleySphere��RastriginRastrigin��AckleyAckley
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',5);  %����matlab�����Ż�������ţ�ٷ���->����Ԥѧϰ�Ż���


%% ��ʼ������
Task = TASK();
Task = initTASK(Task,name);

%% MFEA
%%% 0.��¼���Ž�ľ���
EvBestFitness = zeros(gen+1,Task.M);          %ÿ����õ���Ӧ��ֵ
TotalEvaluations=zeros(gen+1,1);              %ÿ��ÿ���������۴���

%%% 1.��ʼ����Ⱥ
Population = INDIVIDUAL();                    %���ɳ�ʼ��Ⱥ
Population = initPOP(Population,N,Task.D_multitask,Task.M);

%%% 2.���ݶ����񻷾��е�ÿ���Ż���������ÿ����������Ӵ���
[Population,TotalEvaluations(1)] = evaluate(Population,Task,pi_l,options);

%%% 3.�����ʼ����Ⱥ�����صȼ��Լ���������
[Population,EvBestFitness(1,:),bestind] = Calfactor(Population);

%%% 4.�Ż�����
for i = 1:gen
    %4.1 ���彻�����
    Offspring  = GA(Population,rmp,Pc,mu,sigma);
    %4.2 �������Ӵ���
    [Offspring,TotalEvaluations(i+1)] = evaluate(Offspring,Task,pi_l,options);
    TotalEvaluations(i+1) = TotalEvaluations(i+1) + TotalEvaluations(i);
    %4.3 ��Ⱥ�ϲ�
    intpopulation = combpop(Population,Offspring);
    %4.4 ���±�����Ӧ�ȣ��������أ����صȼ�
    [intpopulation,EvBestFitness(i+1,:),bestind] = Calfactor(intpopulation);
    %4.5 ����ѡ��
    Population = EnvironmentalSelection(intpopulation,selection_process,N,Task.M);
    disp(['MFEA Generation = ', num2str(i), ' EvBestFitness = ', num2str(EvBestFitness(i+1,:))]);%Ϊ�˼�¼��ʼ����ֵ���Դ���+1
end

%% ��¼�㷨���
data_MFEA.wall_clock_time=toc;
data_MFEA.EvBestFitness=EvBestFitness;
data_MFEA.bestInd_data=bestind;
data_MFEA.TotalEvaluations=TotalEvaluations;
save(['Data\','data.mat'],'data_MFEA');

%% ��ͼ
for i=1:Task.M
    figure(i)
    hold on
    plot(EvBestFitness(:,i));
    xlabel('GENERATIONS');
    ylabel(['TASK ', num2str(i), ' OBJECTIVE']);
    saveas(gcf,['Data\figure_Task',num2str(i),'.jpg']);
end