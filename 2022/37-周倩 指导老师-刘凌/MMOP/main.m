%% Add path
addpath(genpath('MM_testfunctions/'));
addpath(genpath('Indicators/'));

clear all
clc

global fname  %子函数不传参直接调用主函数global变量方法

N_function=3;  % 测试函数的编号
runtimes=3;  % 实验次数，odd number奇数，方便找中间值
local_function = [10,11,12,13,15,19]; % 这些函数的指标值仅用于全局PS和PF进行计算，意思是第10,11,12,13,15,19个函数有全局PS和PF
t_indicator = zeros(N_function,4);  %均值
d_indicator = zeros(N_function,4);  %标准差
% t_rHV = zeros(N_function,1);
% t_IGDX = zeros(N_function,1);
% t_IGDF = zeros(N_function,1);
%% 初始化测试函数中的参数
for i_func=1:N_function
    switch i_func
        case 1
            fname='MMF1';  % 函数名
            n_obj=2;       % 目标空间的维度
            n_var=2;       % 决策空间的维度
            xl=[1 -1];     % 决策变量的下限
            xu=[3 1];      % 决策变量的上限
            repoint=[1.1,1.1]; % 参考点，方便后续计算HV
        case 2
            fname='MMF2';
            n_obj=2;
            n_var=2;
            xl=[0 0];
            xu=[1 2];
            repoint=[1.1,1.1];
        case 3
            fname='MMF3';
            n_obj=2;
            n_var=2;
            xl=[0 0];
            xu=[1 1.5];
            repoint=[1.1,1.1];
        case 4
            fname='MMF4';
            n_obj=2;
            n_var=2;
            xl=[-1 0];
            xu=[1 2];
            repoint=[1.1,1.1];
        case 5
            fname='MMF5';
            n_obj=2;
            n_var=2;
            xl=[1 -1];
            xu=[3 3];
            repoint=[1.1,1.1];
        case 6
            fname='MMF6';
            n_obj=2;
            n_var=2;
            xl=[1 -1];
            xu=[3 2];
            repoint=[1.1,1.1];
        case 7
            fname='MMF7';
            n_obj=2;
            n_var=2;
            xl=[1 -1];
            xu=[3 1];
            repoint=[1.1,1.1];
        case 8
            fname='MMF8';
            n_obj=2;
            n_var=2;
            xl=[-pi 0];
            xu=[pi 9];
            repoint=[1.1,1.1];
        case 9
            fname='MMF9'; 
            n_obj=2;   
            n_var=2;    
            xl=[0.1 0.1];  
            xu=[1.1 1.1];    
            repoint=[1.21,11]; 
        case 10
            fname='MMF10'; 
            n_obj=2; 
            n_var=2; 
            xl=[0.1 0.1];
            xu=[1.1 1.1]; 
            repoint=[1.21,13.2];
        case 11
            fname='MMF11';
            n_obj=2;
            n_var=2; 
            xl=[0.1 0.1]; 
            xu=[1.1 1.1]; 
            repoint=[1.21,15.4];
        case 12
            fname='MMF12'; 
            n_obj=2; 
            n_var=2;  
            xl=[0 0]; 
            xu=[1 1]; 
            repoint=[1.54,1.1];
        case 13
            fname='MMF13';
            n_obj=2; 
            n_var=3; 
            xl=[0.1 0.1 0.1]; 
            xu=[1.1 1.1 1.1]; 
            repoint=[1.54,15.4];
        case 14
            fname='MMF14'; 
            n_obj=3; 
            n_var=3; 
            xl=[0 0 0]; 
            xu=[1 1 1];
            repoint=[2.2,2.2,2.2];
        case 15
            fname='MMF15'; 
            n_obj=3; 
            n_var=3; 
            xl=[0 0 0]; 
            xu=[1 1 1]; 
            repoint=[2.5,2.5,2.5];
        case 16
            fname='MMF1_z';
            n_obj=2; 
            n_var=2; 
            xl=[1 -1]; 
            xu=[3 1]; 
            repoint=[1.1,1.1];
        case 17
            fname='MMF1_e';
            n_obj=2;
            n_var=2;  
            xl=[1 -20];
            xu=[3 20];
            repoint=[1.1,1.1];
        case 18
            fname='MMF14_a';
            n_obj=3;
            n_var=3;
            xl=[0 0 0];
            xu=[1 1 1];
            repoint=[2.2,2.2,2.2];
        case 19
            fname='MMF15_a';
            n_obj=3;
            n_var=3;
            xl=[0 0 0];
            xu=[1 1 1];
            repoint=[2.5,2.5,2.5];
        case 20
            fname='SYM_PART_simple';
            n_obj=2;
            n_var=2;
            xl=[-20 -20];
            xu=[20 20];
            repoint=[4.4,4.4];
        case 21
            fname='SYM_PART_rotated';
            n_obj=2;
            n_var=2;
            xl=[-20 -20];
            xu=[20 20];
            repoint=[4.4,4.4];
        case 22
            fname='Omni_test';
            n_obj=2;
            n_var=3;
            xl=[0 0 0];
            xu=[6 6 6];
            repoint=[4.4,4.4];
    end
    %% PS 和 PF的载参
    % PS是真实决策空间,也就是变量x；PF是PS在目标空间的映射,也就是目标 f1和 f2
    % 如果当你想用的函数所在的编号 等于 只有全局PF和PF的函数编号
    if sum(i_func == local_function) >0
        load (strcat([fname,'_globalPSPF']))   %加载全局PS PF变量进来，这里只加载全局
        PS = global_PS;
        PF = global_PF;
    else
        load (strcat([fname,'_Reference_PSPF_data']));  % 均为400*2
    end
    %% 初始化种群大小及最大评价次数
    popsize=200; % 即每代的个体数
    Max_fevs=200*100; % Max_fevs是评价次数即适应度即x到y的映射，200个个体，每个个体每一代有一个适应度，100代
    Max_Gen=fix(Max_fevs/popsize);  % fix(X)：将 X 的每个元素朝零方向四舍五入为最近的整数
    Indicator=cell(Max_Gen,1);
    PSdata = cell(Max_Gen,1);
    PFdata = cell(Max_Gen,1);
    for j=1:runtimes %j=1:3
        %% 用NCDE搜索PS. NCDE：首先生成NP个初始解，对于每个个体，找到m个最为相似的个体形成子种群subpopi,在subpopi中找到r1,r2,r3，使用差分进化DE得到ui，计算ui到整个种群其他个体的距离，比较和ui距离最近的个体的适应度，如果ui更优，则取代。相比CDE，NCDE只是加入了领域变异，十分简单。
        fprintf('Running test function: %s, times = %d \n', fname,j);
        [ps,pf]=MMO_DE_CSCD(fname,xl,xu,n_obj,popsize,Max_Gen);
        for g=1:Max_Gen
            g_ps = ps{g};
            g_pf = pf{g};
            % 指标
            hyp=Hypervolume_calculation(g_pf,repoint);  % hv用来评价种群在目标空间中的多样性，但这里用倒数形式（见后面），因此越小越好。
            IGDx=IGD_calculation(g_ps,PS);  % IGD距离越小代表收敛性越好
            IGDf=IGD_calculation(g_pf,PF);
            CR=CR_calculation(g_ps,PS);  % 帕累托集合的覆盖率
            PSP=CR/IGDx; % 加个倒数，rPSP越小越好，不仅反映得到的PSs与真实PSs之间的重叠率，而且反映得到的解的多样性和收敛性。
            Indicator{g, 1}.MMO_DE_CSCD(j,:)=[1./PSP,1./hyp,IGDx,IGDf];  % 存放四个指标结果（3*4），j=1存放第一次实验的指标值在第一行
            
            PSdata{g,1}.MMO_DE_CSCD{j}=g_ps; % j=2，ps放在第2次实验PSdata.MMO_DE_CSCD的第2列
            PFdata{g,1}.MMO_DE_CSCD{j}=g_pf;
            clear hyp IGDx IGDf CR PSP
        end
        clear ps pf
     
        %fprintf('Running test function: %s \n %d times \n', fname,j);
    end
    choose_ps = cell(Max_Gen, 1);
    choose_pf = cell(Max_Gen, 1);
    
    for g=1:Max_Gen
        %% 选择一个具有中位数指标的PS
%         fprintf('%d\n', g);
        choosen_In=Indicator{g}.MMO_DE_CSCD(:,1); % 是MMO_DE_CSCD表第一列，即三次实验得到的rPSP
        median_index=find(choosen_In==median(choosen_In)); % find是找与rPSP三个数算出的中位数0.047与rPSP[0.043 0.047 0.048]本身相等的索引[0 1 0]，median_index是2，即相等的数的索引是2
        choose_ps{g,1}.MMO_DE_CSCD = PSdata{g, 1}.MMO_DE_CSCD{median_index};  % MMO_DE_CSCD第2列（第2次实验得到的ps）
        choose_pf{g,1}.MMO_DE_CSCD = PFdata{g, 1}.MMO_DE_CSCD{median_index};
        clear choosen_In median_index
        
        %% 计算指标的均值和标准差
        Indicator{g, 1}.MMO_DE_CSCD(runtimes+1,:)=min(Indicator{g, 1}.MMO_DE_CSCD(1:runtimes,:)); % 第4行是各指标的最小值，把Indicator.MMO_DE_CSCD前3行各列（指标）的最小值放在第4行对应的列（因为前3行有数据，是3次实验得到的指标）
        Indicator{g, 1}.MMO_DE_CSCD(runtimes+2,:)=max(Indicator{g, 1}.MMO_DE_CSCD(1:runtimes,:)); % 第5行是各指标的最大值
        Indicator{g, 1}.MMO_DE_CSCD(runtimes+3,:)=mean(Indicator{g, 1}.MMO_DE_CSCD(1:runtimes,:)); % 第6行是3次实验各指标的平均值
        Indicator{g, 1}.MMO_DE_CSCD(runtimes+4,:)=median(Indicator{g, 1}.MMO_DE_CSCD(1:runtimes,:)); % 第7行是3次实验各指标的中位数
        Indicator{g, 1}.MMO_DE_CSCD(runtimes+5,:)=std(Indicator{g, 1}.MMO_DE_CSCD(1:runtimes,:)); % 第8行是3次实验各指标的标准差
            
        % 生成表格数据
        Table.MMO_DE_CSCD{i_func}.rPSP(g,:)=(Indicator{g}.MMO_DE_CSCD(:,1))'; % Indicator.MMO_DE_CSCD(8*4)的第1列数据(j次实验的rPSP 最小rPSP 最大rPSP rPSP的平均值 rPSP的中位数 rPSP的标准差)放到Table里
        Table.MMO_DE_CSCD{i_func}.rHV(g,:)=(Indicator{g}.MMO_DE_CSCD(:,2))';
        Table.MMO_DE_CSCD{i_func}.IGDX(g,:)=(Indicator{g}.MMO_DE_CSCD(:,3))';
        Table.MMO_DE_CSCD{i_func}.IGDF(g,:)=(Indicator{g}.MMO_DE_CSCD(:,4))';
        Table.MMO_DE_CSCD{i_func}.rPSP_gen(g,:)=(Indicator{g}.MMO_DE_CSCD(1:runtimes,1))';

    end
    % 保存四个指标21次实验的均值
    t_indicator(i_func,1)= Table.MMO_DE_CSCD{i_func}.rPSP(Max_Gen,runtimes+3);
    t_indicator(i_func,2)= Table.MMO_DE_CSCD{i_func}.rHV(Max_Gen,runtimes+3);
    t_indicator(i_func,3)= Table.MMO_DE_CSCD{i_func}.IGDX(Max_Gen,runtimes+3);
    t_indicator(i_func,4)= Table.MMO_DE_CSCD{i_func}.IGDF(Max_Gen,runtimes+3);
    % 保存四个指标21次实验的标准差
    d_indicator(i_func,1)= Table.MMO_DE_CSCD{i_func}.rPSP(Max_Gen,runtimes+5);
    d_indicator(i_func,2)= Table.MMO_DE_CSCD{i_func}.rHV(Max_Gen,runtimes+5);
    d_indicator(i_func,3)= Table.MMO_DE_CSCD{i_func}.IGDX(Max_Gen,runtimes+5);
    d_indicator(i_func,4)= Table.MMO_DE_CSCD{i_func}.IGDF(Max_Gen,runtimes+5);
    %% 保存结果
    save(strcat([fname,'PSPF_indicator_global_data_MMO_DE_CSCD']),'PSdata','PFdata','Indicator');
    clear PSdata PFdata 
    save(strcat([fname,'ChoosenPSPFdata']),'choose_ps','choose_pf');

    %% 画图
% %     mkdir('C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\');%创建文件夹
    save('C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片');%保存工作区里面的文件
%     
    %---rPSP-----------    
    figure;
    plot(Table.MMO_DE_CSCD{i_func}.rPSP(:, runtimes+3),'b->','MarkerIndices',1:5:length(Table.MMO_DE_CSCD{i_func}.rPSP(:, runtimes+3)));
    ylabel 'rPSP'
    xlabel 'Generation number'
    title(['第',num2str(i_func),'个测试集上的平均rPSP变化图像']);    % 添加图片title
    saveas(gcf, ['C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片\','第', num2str(i_func), '个测试集上的平均rPSP变化图像.jpg']);
    
    %---rHV-----------    
    figure
%     plot(Table.MMO_DE_CSCD.rHV,'r-');
    plot(Table.MMO_DE_CSCD{i_func}.rHV(:, runtimes+3),'b->','MarkerIndices',1:5:length(Table.MMO_DE_CSCD{i_func}.rHV(:, runtimes+3)));
    ylabel 'rHV'
    xlabel 'Generation number'
    title(['第',num2str(i_func),'个测试集上的平均rHV变化图像']);    % 添加图片title
    saveas(gcf, ['C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片\','第', num2str(i_func), '个测试集上的平均rHV变化图像.jpg']);

    %---IGDX-----------    
    figure
    plot(Table.MMO_DE_CSCD{i_func}.IGDX(:, runtimes+3),'b->','MarkerIndices',1:5:length(Table.MMO_DE_CSCD{i_func}.IGDX(:, runtimes+3)));
    ylabel 'IGDX'
    xlabel 'Generation number'    
    title(['第',num2str(i_func),'个测试集上的平均IGDX变化图像']);    % 添加图片title
    saveas(gcf, ['C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片\','第', num2str(i_func), '个测试集上的平均IGDX变化图像.jpg']);

    %---IGDF-----------    
    figure
    plot(Table.MMO_DE_CSCD{i_func}.IGDF(:, runtimes+3),'b->','MarkerIndices',1:5:length(Table.MMO_DE_CSCD{i_func}.IGDF(:, runtimes+3)));
    ylabel 'IGDF'
    xlabel 'Generation number' 
    title(['第',num2str(i_func),'个测试集上的平均IGDF变化图像']);    % 添加图片title
    saveas(gcf, ['C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片\','第', num2str(i_func), '个测试集上的平均IGDF变化图像.jpg']);
    
    %---PF and pf-----------    
    figure
    scatter(PF(:,1), PF(:,2),'o');
    hold on
    scatter(choose_pf{Max_Gen}.MMO_DE_CSCD(:,1),choose_pf{Max_Gen}.MMO_DE_CSCD(:,2),'+');
    ylabel 'f2'
    xlabel 'f1' 
    title(['第',num2str(i_func),'个测试集上的真实PF和得到的pf']);    % 添加图片title
    legend('真实PF','得到的pf');
%     saveas(s1,'C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\PF.png');%保存图片为png格式
%     imwrite(data.image,['image/',num2str(i_func),'.jpg']);
    saveas(gcf, ['C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片\','第', num2str(i_func), '个测试集上的真实PF和得到的pf.jpg']);

    %---PS and ps-----------    
    figure
    scatter(PS(:,1), PS(:,2),'o');
    hold on
    scatter(choose_ps{Max_Gen}.MMO_DE_CSCD(:,1),choose_ps{Max_Gen}.MMO_DE_CSCD(:,2),'+');
    ylabel 'x2'
    xlabel 'x1'
    title(['第',num2str(i_func),'个测试集上的真实PS和得到的ps']);    % 添加图片title
    legend('真实PS','得到的ps');
    saveas(gcf, ['C:\Users\CIA\Desktop\2\1\计算机前沿技术\MMOP\图片\','第', num2str(i_func), '个测试集上的真实PS和得到的ps.jpg']);

end
save Table_global_MMODE_CSCD Table

