% clear all;close all;clc;            %清除所有变量，关闭所有窗口，  清除命令窗口的内容
% x = 0 + (100 - 0).* rand(100, 2);

function APcl = AP(x)
N=size(x,1);              %N为矩阵的列数，即聚类数据点的个数
M=N*N-N;                  %N个点间有M条来回连线，考虑到从i到k和从k到i的距离可能是不一样的
s=zeros(M,3);             %定义一个M行3列的零矩阵,用于存放根据数据点计算出的相似度

j=1;                      %通过for循环给s赋值，第一列表示起点i，第二列为终点k，第三列为i到k的负欧式距离作为相似度
for i=1:N
    for k=[1:i-1,i+1:N]
        s(j,1)=i;s(j,2)=k;
        %         s(j,3)=acos(1-pdist2(x(i,:),x(k,:),'cosine'));
        s(j,3)=-sum((x(i,:)-x(k,:)).^2);
        j=j+1;
    end
end
p=median(s(:,3));           %p为矩阵s第三列的中间值，即所有相似度值的中位数，用中位数作为preference,将获得数量合适的簇的个数
tmp=max(max(s(:,1)),max(s(:,2)));           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S=-Inf*ones(N,N);                           %-Inf负无穷大，定义S为N*N的相似度矩阵，初始化每个值为负无穷大
for j=1:size(s,1)                           %用for循环将s转换为S，S（i，j）表示点i到点j的相似度值
    S(s(j,1),s(j,2))=s(j,3);
end
nonoise=1;                                  %此处仅选择分析无噪情况（即S（i，j）=S（j,i）），所以略去下面几行代码
if length(p)==1                             %设置preference
    for i=1:N
        S(i,i)=p;
    end
else
    for i=1:N
        S(i,i)=p(i);
    end
end
% Allocate space for messages ,etc
dS=diag(S);                                                 %列向量，存放S中对角线元素信息
A=zeros(N,N);
R=zeros(N,N);
%Execute parallel affinity propagation updates
convits=50;maxits=500;                               %设置迭代最大次数为500次，迭代不变次数为50
e=zeros(N,convits);dn=0;i=0;                         %e循环地记录50次迭代信息，dn=1作为一个循环结束信号，i用来记录循环次数
while ~dn
    i=i+1;
    %Compute responsibilities
    Rold=R;                                          %用Rold记下更新前的R
    AS=A+S;                                          %A(i,j)+S(i,j)
    [Y,I]=max(AS,[],2);                              %获得AS中每行的最大值存放到列向量Y中，每个最大值在AS中的列数存放到列向量I中
    for k=1:N
        AS(k,I(k))=-realmax;                         %将AS中每行的最大值置为负的最大浮点数，以便于下面寻找每行的第二大值
    end
    [Y2,I2]=max(AS,[],2);                            %存放原AS中每行的第二大值的信息
    R=S-repmat(Y,[1,N]);                             %更新R,R(i,k)=S(i,k)-max{A(i,k')+S(i,k')}，k'~=k即计算出各点作为i点的簇中心的适合程度
    
    for k=1:N                                        %eg:第一行中AS(1,2)最大,AS(1,3)第二大，
        R(k,I(k))=S(k,I(k))-Y2(k);                   %so R(1,1)=S(1,1)-AS(1,2); R(1,2)=S(1,2)-AS(1,3); R(1,3)=S(1,3)-AS(1,2).............
    end                                              %这样更新R后，R的值便表示k多么适合作为i 的簇中心，若k是最适合i的点，则R(i,k)的值为正
    lam=0.5;
    R=(1-lam)*R+lam*Rold;                            %设置阻尼系数，防止某些情况下出现的数据振荡
    %Compute availabilities
    Aold=A;
    Rp=max(R,0);                                     %除R(k,k)外，将R中的负数变为0，忽略不适和的点的不适合程度信息
    for k=1:N
        Rp(k,k)=R(k,k);
    end
    A=repmat(sum(Rp,1),[N,1])-Rp;                    %更新A(i,k),先将每列大于零的都加起来，因为i~=k,所以要减去多加的Rp(i,k)
    dA=diag(A);
    A=min(A,0);                         %除A(k,k)以外，其他的大于0的A值都置为0
    for k=1:N
        A(k,k)=dA(k);
    end
    A=(1-lam)*A+lam*Aold;               %设置阻尼系数，防止某些情况下出现的数据振荡
    %Check for convergence
    E=((diag(A)+diag(R))>0);
    e(:,mod(i-1,convits)+1)=E;               %将循环计算结果列向量E放入矩阵e中，注意是循环存放结果，即第一次循环得出的E放到N*50的e矩阵的第一列，第51次的结果又放到第一列
    K=sum(E);                                %每次只保留连续的convits条循环结果，以便后面判断是否连续迭代50次中心簇结果都不变。%%%%%%%%%%%%%%%%
    if i>=convits || i>=maxits               %判断循环是否终止
        se=sum(e,2);                         %se为列向量，E的convits次迭代结果和
        unconverged=(sum((se==convits)+(se==0))~=N);%所有的点要么迭代50次都满足A+R>0，要么一直都小于零，不可以作为簇中心
        if (~unconverged&&(K>0))||(i==maxits) %迭代50次不变，且有簇中心产生或超过最大循环次数时循环终止。
            dn=1;
        end
    end
end
I=find(diag(A+R)>0);                        %I是每个簇的中心；； %经过上面的循环，便确定好了哪些点可以作为簇中心点，用find函数找出那些簇1中心点,这个简单demo中I=[2,4],
K=length(I); % Identify exemplars                                                                                                           %即第二个点和第四个点为这六个点的簇中心
if K>0                                      %如果簇中心的个数大于0
    [~,c]=max(S(:,I),[],2);                 %取出S中的第二，四列；求出2，4列的每行的最大值，如果第一行第二列的值大于第一行第四列的值，则说明第一个点是第二个点是归属点
    c(I)=1:K;                               % Identify clusters c(2)=1,c(4)=2(第2个点为第一个簇中心，第4个点为第2个簇中心)
    % Refine the final set of exemplars and clusters and return results
    for k=1:K
        ii=find(c==k);                      %k=1时，发现第1，2，3个点为都属于第一个簇
        [y,j]=max(sum(S(ii,ii),1));         %k=1时 提取出S中1，2，3行和1，2，3列组成的3*3的矩阵，分别算出3列之和取最大值，y记录最大值，j记录最大值所在的列
        I(k)=ii(j(1));                      %I=[2;4]
    end
    [tmp,c]=max(S(:,I),[],2);        %tmp为2，4列中每行最大数组成的列向量，c为每个最大数在S（：，I）中的位置，即表示各点到那个簇中心最近
    c(I)=1:K;                        %c(2)=1;c(4)=2;
    tmpidx=I(c);                     %I=[2;4],c中的1用2替换，2用4替换
    %(tmpidx-1)*N+(1:N)'                    %一个列向量分别表示S(1,2),S(2,2),S(3,2),S(4,4),S(5,4),S(6,4)是S矩阵的第几个元素
    %sum(S((tmpidx-1)*N+(1:N)'))            %求S中S(1,2)+S(2,2)+S(3,2)+S(4,4)+S(5,4)+S(6,4)的和
    tmpnetsim=sum(S((tmpidx-1)*N+(1:N)'));  %将各点到簇中心的一个表示距离的负值的和来衡量这次聚类的适合度
    tmpexpref=sum(dS(I));                   %dS=diag(S)；               %表示所有被选为簇中心的点的适合度之和
else
    tmpidx=nan*ones(N,1);  %nan Not A Number 代表不是一个数据。数据处理时，在实际工程中经常数据的缺失或者不完整，此时我们可以将那些缺失设置为nan
    tmpnetsim=nan;
    tmpexpref=nan;
end



netsim=tmpnetsim;                           %反应这次聚类的适合度
dpsim=tmpnetsim-tmpexpref;                  %数据点与其样本的相似度之和
expref=tmpexpref;                           %已识别示例的偏好之和
idx=tmpidx;                                 %记录了每个点所属那个簇中心的列向量
unique(idx);

h = 1;
for i=unique(idx)'
    ii=find(idx==i);
    APcl{h} = ii';
    h = h + 1;
    
end
end

