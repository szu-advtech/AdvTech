function [Indicator,rep]=MMODE_ICD(fname,n_var,n_obj,xl,xu,method,feval_max,Particle_Number,PS,PF,repoint)
% MMODE_ICD: Differential Evolution Using Improved Crowding Distance for Multimodal Multiobjective Optimization,Swarm and Evolutionary Computation. vol. 62, pp. 100849: 1-10, 2021. 
%==============
% Caitong Yue  Aug. 2019  
%================
%================================================
% st:  learning strategy of DE
% F:   scaling factor
% CR:  crossover rate
% feval_max: the maximum FEs
% max_rep_size: the maximum archive size
%================================================
%% Input:
%                      Dimension                    Description
%      fname           1 x length function name     the name of test function     
%      n_var           1 x 1                        dimensions of decision space
%      n_obj           1 x 1                        dimensions of objective space
%      xl              1 x n_var                    low bound of decision variable
%      xu              1 x n_var                    up bound of decision variable
%      method          1 x 1                        DE operation: DE/rand/2
%      feval_max       1 x 1                        Maximum fitness evaluations
%      Particle_Number 1 x 1                        population size
%      PS              Particle_Number x n_var      Ture PS to calculate performance indicators
%      PF              Particle_Number x n_obj      Ture PF to calculate performance indicators
%      repoint         1 x n_obj                    repoint
%% Output:
%                     Description
%      ps             Pareto set
%      pf             Pareto front
%%  Reference and Contact 
% Reference: C. T. Yue, P. N. Suganthan, J. J. Liang, B. Y. Qu, K. J. Yu, Y. S. Zhu and L. Yan, “Differential Evolution Using Improved Crowding Distance for Multimodal Multiobjective Optimization,” Swarm and Evolutionary Computation. vol. 62, pp. 100849: 1-10, 2021. 
% Contact: For any questions, please feel free to send email to zzuyuecaitong@163.com.

f_1 = 1;
f_2 = 1;
st=method;
F=0.5;       % 差分进化缩放因子
CR=0.1;      % 差分进化交叉概率 
maxgen=ceil(feval_max/Particle_Number);
  
feval_count=0;

    for i=1:Particle_Number     
        parent(i,1:n_var)=xl+(xu-xl).*rand(1,n_var);             % Initialize the population
    end

    for i=1:Particle_Number
        feval_count=feval_count+1;
        parent(i,n_var+1:n_var+n_obj)=feval(fname,parent(i,1:n_var));% Evaluate the population and put fit to the end 
    end
    
    
rep=parent;
rep_comb=parent;

archive=[];
for gcount=1:maxgen
        rep=unique(rep,'rows','stable');
        parent=unique(parent,'rows','stable');
    for j=1:size(rep,1)
        bm=rep(1,1:n_var);% **** change the above two lines to this line, because bm need to be the best one not the random one
        popold=rep(j,1:n_var);%current particle to generate offsprings with DE
        newpop=rep;%the population to calculate distance to the current particle
        poptemp=repmat(parent(j,n_var+1:n_var+n_obj),size(newpop,1),1);
        distance_obj=sqrt(sum((newpop(:,n_var+1:n_var+n_obj)-poptemp(:,1:n_obj)).*(newpop(:,n_var+1:n_var+n_obj)-poptemp(:,1:n_obj)),2));
        [bb,cc]=sort(distance_obj);
         n_zero_obj=sum(distance_obj==0);
         N_size_OS=12; % DE/rand/2策略需要五个个体，从邻居而不是从整个群体中选择
         newpop1=newpop(cc(n_zero_obj+1:n_zero_obj+N_size_OS),:);%the first one is exactly the j_th solution, so start from 2
         tem_newpop1=newpop1(:,1:n_var);
         [~,ia,~]= unique(tem_newpop1,'rows','stable');
         newpop1=newpop1(ia,:);
         cc1=randperm(size(newpop1,1)); % randperm(m, n) 从前 m 个数中随机选择 n 个数，并将其顺序打乱
      % calculate the crowding distance in distance space
        poptemp2=repmat(parent(j,1:n_var),size(newpop1,1),1);
        distance_var=sqrt(sum((newpop1(:,1:n_var)-poptemp2(:,1:n_var)).*(newpop1(:,1:n_var)-poptemp2(:,1:n_var)),2));
        [~,cc2]=sort(distance_var);

% randomly  tournament selection
       P_tournament=rand;
       p1=1-(gcount-1)/maxgen;
       p2=(1-p1)/2;
       p3=1-p1-p2;
       if P_tournament<p1  %Random selection from 5 randomly selected OS neighbours
           newpop2=newpop1(cc1(1:5),:);
       elseif  P_tournament<p1+p2  %Based on larger DS crowding among 5 DS neighbours among the OS neighbours
            if gcount==1
                newpop2=newpop1(cc2(1:5),:);%
            else
                tem_pm1=newpop1(cc2(1:5),:);
                [~,sort_index]=sort(tem_pm1(:,n_var+n_obj+2),'descend');%CD_var
                pm1=tem_pm1(sort_index(1),:);
                pm2_5=setdiff(tem_pm1,pm1,'rows');
                newpop2=[pm1;pm2_5];
            end
       else  %Based on larger OS crowding among 5 randomly selected OS neighbours
            if gcount==1
                newpop2=newpop1(cc1(1:5),:); %newpops are particles near to current particle in both decision and objective space
            else
                tem_pm1=newpop1(cc1(1:5),:);
                [~,sort_index]=sort(tem_pm1(:,n_var+n_obj+3),'descend');%CD_obj
                pm1=tem_pm1(sort_index(1),:);
                pm2_5=setdiff(tem_pm1,pm1,'rows');
                newpop2=[pm1;pm2_5];
            end
       end
        child(j,1:n_var)=DE(popold,newpop2,bm,st,F,CR,n_var,size(newpop2,1),xl,xu);% generate offsprings with DE
        child(j,n_var+1:n_var+n_obj)=feval(fname,child(j,1:n_var));% evaluate the offsprings
        feval_count=feval_count+1;
    end 
    
    rep=rep(:,1:n_var+n_obj);
    if gcount>1
        archive=archive(:,1:n_var+n_obj);%--
    end
    
    rep_comb=[rep;child;archive];%--

    Restrict_rate=0.5;
    Ending_ponit=1;
    Rate_F1=min(Restrict_rate+((1-Restrict_rate)/(Ending_ponit*maxgen))*(gcount-1),1);% the rate selected in rank1 increase from 0.5~1 in generation 1~0.8max_gen  
    Num_sele=Particle_Number;% the number of particles selected from the sorted population
    [rep,tem_archive]=non_domination_combine_cd_sort(rep_comb,n_obj,n_var,Rate_F1,Num_sele);
    %---add archive
        n_tem_arc=size(tem_archive,1);
        if  n_tem_arc>200
            archive=tem_archive(1:200,:);
        else 
            archive=tem_archive;
        end
    %-end--add archive
    if size(rep,1)>Particle_Number
        rep=rep(1:Particle_Number,:);
    end
    parent=rep;
 %-----record the indicators in each generation--------
     ps=rep(:,1:n_var);
     pf=rep(:,n_var+1:n_var+n_obj);
     
         %-----record the indicators in each generation--------
    ps=rep(:,1:n_var);
    pf=rep(:,n_var+1:n_var+n_obj);
    
    %% 画图
    if(mod(gcount,5) == 0)
        %% 画决策空间图
        figure(1)
        subplot(2,1,1)
        delete(f_1)
        if(n_var == 2)
            scatter(PS(:,1),PS(:,2),'b');
            hold on
            f_1 = scatter(ps(:,1),ps(:,2),'r');
        elseif(n_var == 3)
            scatter3(PS(:,1),PS(:,2),PS(:,3),'b');
            hold on
            f_1 = scatter3(ps(:,1),ps(:,2),ps(:,3),'red');
        end
        hold on 

        %% 画目标空间图
    %     figure(2)
        subplot(2,1,2)
        delete(f_2)
        if(n_obj == 2)
            scatter(PF(:,1),PF(:,2),'b');
            hold on
            f_2 = scatter(pf(:,1),pf(:,2),'r');
        elseif(n_obj == 3)
            scatter3(PF(:,1),PF(:,2),PF(:,3),'b');
            hold on
            f_2 = scatter3(pf(:,1),pf(:,2),pf(:,3),'r');
        end
    end
    
     
   %% Indicators
     hyp=Hypervolume_calculation(pf,repoint);
     IGDx=IGD_calculation(ps,PS);
     IGDf=IGD_calculation(pf,PF);
     CR=CR_calculation(ps,PS,fname);
     PSP=CR/IGDx;%
     Indicator(gcount,:)=[IGDf,1./hyp,IGDx,1./PSP];
  %----------
%     disp([num2str(gcount) '/' num2str(maxgen)])
   
    if feval_count>feval_max
        break, 
    end
%    Cluster_x=adj_mat_clustering(rep(:,1:2),0,0.35,3);
    
end
end

function [selected_pop,deleted_pop]=non_domination_combine_cd_sort(x,n_obj,n_var,Rate_F1,Num_sele)
% computing crowding distance of new possible next generation patents considering already selected parents for next generation
% Input  x            the points need to be ranked: popsize*(n_var+n_obj)
%        n_obj        number of objective 
%        n_var        number of varibles
%        Rate_F1      the rate selected particles in the first rank
%        Num_sele     the number of particles need to be selected from x
% Output sorted_based_on_front      the rank value is added to the
%     P_selected      the selected population

    [N_particle, ~] = size(x);% Obtain the number of particles
    front = 1;% Initialize the front number to 1.
    F(front).f = [];
    individual = [];

%% Non-Dominated sort. 

    for i = 1 : N_particle
        % Number of individuals that dominate this individual
        individual(i).n = 0; 
        % Individuals which this individual dominate
        individual(i).p = [];
        for j = 1 : N_particle
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k = 1 : n_obj
                if (x(i,n_var + k) < x(j,n_var + k))
                    dom_less = dom_less + 1;
                elseif (x(i,n_var + k) == x(j,n_var + k))  
                    dom_equal = dom_equal + 1;
                else
                    dom_more = dom_more + 1;
                end
            end
            if dom_less == 0 && dom_equal ~= n_obj
                individual(i).n = individual(i).n + 1;
            elseif dom_more == 0 && dom_equal ~= n_obj
                individual(i).p = [individual(i).p j];
            end
        end   
        if individual(i).n == 0
            x(i,n_obj + n_var + 1) = 1;
            F(front).f = [F(front).f i];
            num_in_front(front,1)=length(F(front).f);% the number of particles in each front 
        end
    end
% Find the subsequent fronts
    while ~isempty(F(front).f)
       Q = [];
       for i = 1 : length(F(front).f)
           if ~isempty(individual(F(front).f(i)).p)
                for j = 1 : length(individual(F(front).f(i)).p)
                    individual(individual(F(front).f(i)).p(j)).n = ...
                        individual(individual(F(front).f(i)).p(j)).n - 1;
                    if individual(individual(F(front).f(i)).p(j)).n == 0
                        x(individual(F(front).f(i)).p(j),n_obj + n_var + 1) = ... 
                            front + 1;
                        Q = [Q individual(F(front).f(i)).p(j)];
                    end
               end
           end
       end
       front =  front + 1;
       F(front).f = Q;
       num_in_front(front,1)=length(F(front).f);% the number of particles in each front 
    end
    % Identify which front we selected to
        selected_num=0;
        front_i=0;
        while selected_num<Num_sele
            front_i=front_i+1;
            if front_i>size(num_in_front,1)
                front_i=front_i-1;
                break
            end
            selected_num=selected_num+fix(Rate_F1*num_in_front(front_i,1));% only select part of Front1
%             if front_i==1
%                 selected_num=selected_num+fix(Rate_F1*num_in_front(front_i,1));% only select part of Front1
%             else
%                 selected_num=selected_num+num_in_front(front_i,1);
%             end
        end
        Max_s_front=front_i;%the max seleted front number
    % Sort the population according to the front number
        [~,index_of_fronts] = sort(x(:,n_obj + n_var + 1));
        for i = 1 : length(index_of_fronts)
            sorted_based_on_front(i,:) = x(index_of_fronts(i),:);
        end 
    current_index = 0;
%%  Crowding Distance
    selected_pop=[];
    deleted_pop=[];
    for front = 1 : Max_s_front
        crowd_dist_obj = 0;
        y = [];
        previous_index = current_index + 1;
        for i = 1 : length(F(front).f)
            y(i,:) = sorted_based_on_front(current_index + i,:);%put the front_th rank into y
        end
        current_index = current_index + i;
   % Sort each individual based on the objective
        if front>1
             y=[y;selected_pop(:,1:n_obj + n_var + 1)];% add the selected pop into current front
        end
        for i = n_var+1 : n_obj+n_var
            [~, index_of_objectives] = sort(y(:,i));
            sorted_based_on_objective = [];
            for j = 1 : length(index_of_objectives)
                sorted_based_on_objective(j,:) = y(index_of_objectives(j),:);
            end
            f_max = ...
                sorted_based_on_objective(length(index_of_objectives), i);
            f_min = sorted_based_on_objective(1,  i);

            if length(index_of_objectives)==1
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;  %If there is only one point in current front

            else
                % deal with boundary points in both decision and objective space
                % twice the distance between the boundary points and its nearest neibohood 
                if (f_max - f_min == 0) % only one point in the current Front
                    y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)=rand;
                    y(index_of_objectives(1),n_obj + n_var + 1 + i)=rand;
                else
                    y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)...
                        = 2*(sorted_based_on_objective(length(index_of_objectives), i)-...
                    sorted_based_on_objective(length(index_of_objectives) -1, i))/(f_max - f_min);
                     y(index_of_objectives(1),n_obj + n_var + 1 + i)=2*(sorted_based_on_objective(2, i)-...
                    sorted_based_on_objective(1, i))/(f_max - f_min);
                end
            end
             for j = 2 : length(index_of_objectives) - 1
                next_obj  = sorted_based_on_objective(j + 1, i);
                previous_obj  = sorted_based_on_objective(j - 1,i);
                if (f_max - f_min == 0) % only one point in the current Front
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = rand;
                else
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = ...
                         (next_obj - previous_obj)/(f_max - f_min);
                end
             end
        end
   
    %% Calculate distance in objective space
        crowd_dist_obj = [];
        crowd_dist_obj(:,1) = zeros(size(y,1),1);
        for i = 1 : n_obj
            crowd_dist_obj(:,1) = crowd_dist_obj(:,1) + y(:,n_obj + n_var + 1+n_var + i);
        end

        avg_crowd_dist_obj=mean(crowd_dist_obj);

        y = y(:,1 : n_obj + n_var+2 );
    
 
        %% calculate the distance in decision sapce
        crowd_dist_var = [];
            newpop=y;
%             k1=4;  %%%%%%%%%%%%%%%%%%%%%%%%%%
            k1=ceil(0.2*prod(max(y(:,1:n_var))-min(y(:,1:n_var))).^(1./n_var)); % 邻居最长的长度
            for p_i=1:size(y,1)
             poptemp=repmat(y(p_i,1:n_var),size(newpop,1),1);
        %      distance(:,p_i)=sqrt(sum((newpop(:,1:n_var)-poptemp(:,1:n_var)).*(newpop(:,1:n_var)-poptemp(:,1:n_var)),2));%the first column is distance from all particle to the first particle
             distance=sqrt(sum((newpop(:,1:n_var)-poptemp(:,1:n_var)).*(newpop(:,1:n_var)-poptemp(:,1:n_var)),2));% popsize*1
             [sorted_dis,ind_dis]=sort(distance');% sorting the distance to p_i th particle in desend order
             
%                  num = sum(distance<k1);
%                  crowd_dist_var(p_i,1)=sum([1:num].*sorted_dis(2:num+1));
             if length(distance)>k1   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                 crowd_dist_var(p_i,1)=sum([1:k1].*sorted_dis(2:k1+1));%Eq.(8) in Combining Crowding Estimation in Objective and Decision Space With Multiple Selection and Search Strategies for Multi-Objective Evolutionary Optimization

             else
                 crowd_dist_var(p_i,1)=sum([2:length(distance)].*sorted_dis(2:end));% the

             end
             
             
            end
            crowd_dist_var=crowd_dist_var./max(crowd_dist_var);
            avg_crowd_dist_var=mean(crowd_dist_var);
            y(:,n_obj+n_var+3)=crowd_dist_var;%put the crowding distace in the decision space in the column (n_obj+n_var+3)



            
         %% Calculate special crowding distance
            special_crowd_dist=zeros(size(crowd_dist_obj,1),1);
           
            for i_SCD = 1 : size(crowd_dist_obj,1)
                    if crowd_dist_obj(i_SCD)>avg_crowd_dist_obj||crowd_dist_var(i_SCD)>avg_crowd_dist_var
                        special_crowd_dist(i_SCD)=max(crowd_dist_obj(i_SCD)/front,crowd_dist_var(i_SCD)); % Eq. (6) in the paper
                    else
                        special_crowd_dist(i_SCD)=min(crowd_dist_obj(i_SCD),crowd_dist_var(i_SCD)); % Eq. (7) in the paper
                    end
            end
           
            y(:,n_obj+n_var+4)=special_crowd_dist;%put the crowding distace in the decision space in the column (n_obj+n_var+4)
            y=y(1:num_in_front(front,1),:);%Only take out the particles in the current front
            [~,index_sorted_based_SCD]=sort(y(:,n_obj+n_var+4),'descend');%sort the particles in the same front according to SCD
            y=y(index_sorted_based_SCD,:);
            selected_pop=[selected_pop;y(1:fix(Rate_F1*num_in_front(front,1)),:)];
            tem_dele=setdiff(y,y(1:fix(Rate_F1*num_in_front(front,1)),:),'rows');
            deleted_pop=[deleted_pop;tem_dele];
             
    end
    
end

function ui=DE(popold,pop,bm,st,F,CR,n,NP,xl,xu)
%% Input
%  popold  the old population
%  pop     the pop to generate difference vectors. in this algorithm they are close to current individual in objectivespace. Generally speaking, they are excellent individuls
%  bm      an individual randomly selected from the offsprings
%  st      the DE opration method 
%  F       the factor before the difference vectors
%  CR      the crossover rate
%  n       the number of decision variables
%  NP      the population size of pop


r1=1;r2=2;r3=3;r4=4;r5=5;

pm1=pop(r1,1:n);
pm2=pop(r2,1:n);
pm3=pop(r3,1:n);
pm4=pop(r4,1:n);
pm5=pop(r5,1:n);
rotd= (0:1:n-1); 

mui = rand(1,n) < CR;          % all random numbers < CR are 1, 0 otherwise
if mui==zeros(1,n),nn=randperm(n);mui(nn(1))=1;end
if st>5
    st=st-5;
    mui=sort(mui');
    nn=floor(rand.*n);
    if nn>0
        rtd = rem(rotd+nn,n);
        mui(:) = mui(rtd+1);  %rotate column i by n
    end
    mui=mui';
end
mpo = mui < 0.5;                % inverse mask to mui

if (st == 1)                % DE/best/1   6
    ui = bm + F*(pm1 - pm2);        % differential variation
    ui = popold.*mpo + ui.*mui;     % binomial crossover
elseif (st == 2)                  % DE/rand/1   7
    ui = pm3 + F*(pm1 - pm2);       % differential variation
    ui = popold.*mpo + ui.*mui;     % crossover
elseif (st == 3)                  % DE/rand-to-best/1    8
    ui = popold + F*(bm-popold) + F*(pm1 - pm2);        
    ui = popold.*mpo + ui.*mui;     % crossover
elseif (st == 4)                  % DE/best/2           9
    ui = bm + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
    ui = popold.*mpo + ui.*mui;           % crossover
elseif (st == 5)                  % DE/rand/2           10
    ui = pm5 + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
    ui = popold.*mpo + ui.*mui;            % crossover
end

ui=(ui<xl).*xl+(ui>=xl).*ui;
ui=(ui>xu).*xu+(ui<=xu).*ui;
end
%==============
% MMODE_ICD: Differential Evolution Using Improved Crowding Distance for Multimodal Multiobjective Optimization,Swarm and Evolutionary Computation. vol. 62, pp. 100849: 1-10, 2021.
%==============
% Caitong Yue  Aug. 2019  
%================