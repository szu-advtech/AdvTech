function [P,LabelSet,ideal,nadir] = classification(Population,ideal,nadir)


%%通过非支配排序把种群一分为二
  [N,M] = size(Population.objs);
  [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
  front = unique(FrontNo)';
  %%更新ideal和nadir
  %有一个问题，特别差的解怎么归一化
     next = zeros(1,N);
     next0 = find(FrontNo<MaxFNo);
     if(sum(next0)==0)
         next0 =  find(FrontNo==MaxFNo);
     end
     
     next(next0) = true;
     p = Population(logical(next')).objs;
     ideal = min(ideal,min(p,[],1));
     nadir= max(nadir,max(p,[],1));
  remain =N-fix(N/2);
  i = 1;
  q = sum(FrontNo==front(i));
  s = [remain>0,remain>=q];
  Next = zeros(1,N);
  Next1 = zeros(1,N);
  while sum(s)==2
    Next0 = find(FrontNo==front(i));
    Next(Next0) = true;
    remain =remain -sum(FrontNo==front(i));
    i = i+1;
    q = sum(FrontNo==front(i));
    s = [remain >0,remain >=q];
  end
  if(remain >0)
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    Last = find(FrontNo==front(i));
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:remain))) = true;
    Next1(Last(Rank(remain+1:q))) = true;
  end
  i = i+1;
  %%问题4索引超出
%   while front(i)~=0
%      Next1(find(FrontNo==front(i))) =true; 
%      i=i+1;
% %      if sum(Next1)==(fix(N/2))
% %          break;
% %      end
%    if i-1 ==MaxFNo
%     break;
%    end
%   end
    Next1(find(Next==0))=true;
  E = Population(logical(Next'));
  P =Population(logical(Next1'));
  
  %%两个解集配对
  
   LabelSet = getLabel1(E,P,ideal,nadir);
%   LabelSet = getLabel(E,P,ideal,nadir);
end
   
    function LabelSet = getLabel(E,P,ideal,nadir)
         
         P1 = P.objs;
         E1 = E.objs;
         [N1,M1] = size(P1);
         [N2,~] = size(E1);
  
         P1 =(P1 - repmat(ideal,N1,1))./(repmat(nadir,N1,1)-repmat(ideal,N1,1));
         E1 =(E1 - repmat(ideal,N2,1))./(repmat(nadir,N2,1)-repmat(ideal,N2,1));
%          %%避免0/0的情况
         P1(find(isnan(P1)))=0.000001;
         E1(find(isnan(E1)))=0.000001;

         Next2 = zeros(1,fix(N2));
        for i=1:N1 
             dis=distance(P1(i,:),E1); 
             index=find(dis==min(dis,[],2));
             Next2(i)=index(1);
        end
         LabelSet = E(logical(Next2'));

    end
       function LabelSet = getLabel1(E,P,ideal,nadir)
         
         P1 = P.objs;
         E1 = E.objs;
         [N1,M1] = size(P1);
         [N2,~] = size(E1);
         
        for i = 1:N1
            rad(i) = randperm(N2,1);     
        end
        
        LabelSet = E(rad(i));
    end
    %公式对不对
    function dis=distance(p,E1)
        [N2,~] = size(E1);
        sum1 = sum(p);
        v1 = p/sum1;
        dis = zeros(1,N2);
        for j= 1:N2
            e = E1(j,:);
            sum2 = sum(e);
            v2 = e/sum2;
            dis(j) = sum((v1-v2).^2);
        end
     
    end
 
  
  