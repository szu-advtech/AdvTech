%% Add path

addpath(genpath('MM_testfunctions/'));
addpath(genpath('Indicator_calculation/'));
 
clear all
 clc

  global fname
  N_function=22;% number of test function
  runtimes=21;% odd number
  fun_num=3;%function number
  local_function = [10,11,12,13,15,19];
 %% Initialize the parameters in MMO test functions
     %[2 3]
     for i_func= fun_num
        switch i_func
            case 1
                fname='MMF1';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[1 -1];     % the low bounds of the decision variables
                xu=[3 1];      % the up bounds of the decision variables
                repoint=[1.1,1.1]; % reference point used to calculate the Hypervolume, it is set to 1.1*(max value of f_i)     
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
                fname='MMF9';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[0.1 0.1];     % the low bounds of the decision variables
                xu=[1.1 1.1];      % the up bounds of the decision variables
                repoint=[1.21,11]; % reference point used to calculate the Hypervolume
            case 10
               fname='MMF10';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[0.1 0.1];     % the low bounds of the decision variables
                xu=[1.1 1.1];      % the up bounds of the decision variables
               repoint=[1.21,13.2]; % reference point used to calculate the Hypervolume
            case 11
                fname='MMF11';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[0.1 0.1];     % the low bounds of the decision variables
                xu=[1.1 1.1];      % the up bounds of the decision variables
                repoint=[1.21,15.4];
            case 12
                fname='MMF12';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[0 0];     % the low bounds of the decision variables
                xu=[1 1];      % the up bounds of the decision variables
                repoint=[1.54,1.1];
             case 13
                 %*need to be modified
                fname='MMF13';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=3;       % the dimensions of the objective space
                xl=[0.1 0.1 0.1];     % the low bounds of the decision variables
                xu=[1.1 1.1 1.1];      % the up bounds of the decision variables
                repoint=[1.54,15.4];
             case 14
                fname='MMF14';  % function name
                n_obj=3;       % the dimensions of the decision space
                n_var=3;       % the dimensions of the objective space
                xl=[0 0 0];     % the low bounds of the decision variables
                xu=[1 1 1];      % the up bounds of the decision variables
                repoint=[2.2,2.2,2.2];
              case 15
                fname='MMF15';  % function name
                n_obj=3;       % the dimensions of the decision space
                n_var=3;       % the dimensions of the objective space
                xl=[0 0 0];     % the low bounds of the decision variables
                xu=[1 1 1];      % the up bounds of the decision variables
                repoint=[2.5,2.5,2.5];
             case 16
                fname='MMF1_z';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[1 -1];     % the low bounds of the decision variables
                xu=[3 1];      % the up bounds of the decision variables
                repoint=[1.1,1.1];
            case 17
                fname='MMF1_e';  % function name
                n_obj=2;       % the dimensions of the decision space
                n_var=2;       % the dimensions of the objective space
                xl=[1 -20];     % the low bounds of the decision variables
                xu=[3 20];      % the up bounds of the decision variables
                repoint=[1.1,1.1];
            case 18
                fname='MMF14_a';  % function name
                n_obj=3;
                n_var=3;
                xl=[0 0 0];
                xu=[1 1 1];
                repoint=[2.2,2.2,2.2];
            case 19
                fname='MMF15_a';  % function name
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
       %% Load reference PS and PF data
       if ismember(i_func, local_function)
           load (strcat([fname,'_globalPSPF']));
           PS = global_PS;
           PF = global_PF;
       else
           % 加载PS PF变量， 均为400*2 double
           % PS是决策空间,也就是变量x;PF是PS在目标空间的映射,也就是目标 f1和 f2
           load  (strcat([fname,'_Reference_PSPF_data']));
       end
%           load  (strcat([fname,'_Reference_PSPF_data']));
       %% Initialize the population size and the maximum evaluations
          popsize=100*n_var;
          Max_fevs=5000*n_var;
          Max_Gen=fix(Max_fevs/popsize);
          method=5;%choose which method in DE
           for j=1:runtimes
               %% Search the PSs using MMODE
               fprintf('Running test function: %s \n %d times \n', fname,j);
                   [Indicator_MMODE_ICD(:,:,j),rep_MMODE_ICD]=MMODE_ICD(fname,n_var,n_obj,xl,xu,method,Max_fevs,popsize,PS,PF,repoint);
                  
              
           end
     end 
      
        figure
        subplot(2,1,1)
     %% 画决策空间图
        if(n_var == 2)
            scatter(PS(:,1),PS(:,2),'b');
            hold on
            scatter(rep_MMODE_ICD(:, 1) ,rep_MMODE_ICD(:, 2),'r');
            legend('True PS', 'Obtain PS');
            xlabel 'x1'
            ylabel 'x2'
        elseif(n_var == 3)
            scatter3(PS(:,1),PS(:,2),PS(:,3),'b');
            hold on
            scatter3(rep_MMODE_ICD(:, 1) ,rep_MMODE_ICD(:, 2),rep_MMODE_ICD(:, 3),'r');
            legend('True PS', 'Obtain PS');
            xlabel 'x1'
            ylabel 'x2'
            zlabel 'x3'
        end

        %% 画目标空间图
        subplot(2,1,2)
        if(n_obj == 2)
            scatter(PF(:,1),PF(:,2),'b');
            hold on
            scatter(rep_MMODE_ICD(:, n_var+1) ,rep_MMODE_ICD(:, n_var+2),'r');
            legend('True PF', 'Obtain PF');
            xlabel 'f1'
            ylabel 'f2'
        elseif(n_obj == 3)
            scatter3(PF(:,1),PF(:,2),PF(:,3),'b');
            hold on
            scatter3(rep_MMODE_ICD(:, n_var+1) ,rep_MMODE_ICD(:, n_var+2),rep_MMODE_ICD(:, n_var+3),'r');
            legend('True PF', 'Obtain PF');
            xlabel 'f1'
            ylabel 'f2'
            zlabel 'f3'
        end
     
        
      %% calculate the mean value
        Mean_Indicator_MMODE_ICD=mean(Indicator_MMODE_ICD,3);
          
       %---IDG-----------  
          figure   
          subplot(2, 2, 1)
          plot(1:size(Mean_Indicator_MMODE_ICD,1),Mean_Indicator_MMODE_ICD(:,1),'r-');
          ylabel 'IGD'
          xlabel 'Generation number'
          
       %---IDGx-----------  
%           figure
          subplot(2, 2, 2)
          plot(1:size(Mean_Indicator_MMODE_ICD,1),Mean_Indicator_MMODE_ICD(:,3),'r-');
          ylabel 'IGDx'
          xlabel 'Generation number'
          
       %---rPSP-----------    
%           figure
          subplot(2, 2, 4)
          plot(1:size(Mean_Indicator_MMODE_ICD,1),Mean_Indicator_MMODE_ICD(:,4),'r-');
          ylabel 'rPSP'
          xlabel 'Generation number'
          
       %---rHV-----------    
%           figure
          subplot(2, 2, 3)
          plot(1:size(Mean_Indicator_MMODE_ICD,1),Mean_Indicator_MMODE_ICD(:,2),'r-');
          ylabel 'rHV'
          xlabel 'Generation number'