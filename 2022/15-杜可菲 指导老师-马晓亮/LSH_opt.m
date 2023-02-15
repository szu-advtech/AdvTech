function X = lhsdesign(n,p,varargin)
if nargin > 2
    [varargin{:}] = convertStringsToChars(varargin{:});
end

okargs = {'iterations' 'criterion' 'smooth'};
defaults = {NaN 'maximin' 'on'};
[maxiter,crit,dosmooth] = internal.stats.parseArgs(okargs,defaults,varargin{:});

if isempty(maxiter)
    maxiter = NaN;
elseif ~isnumeric(maxiter) || ~isscalar(maxiter) || maxiter<0
    error(message('stats:lhsdesign:ScalarRequired'));
end
if isnan(maxiter), maxiter = 5; end

okcrit = {'none' 'maximin' 'correlation'};
crit = internal.stats.getParamVal(crit,okcrit,'Criterion');

if isempty(dosmooth)
    dosmooth = 'on';
elseif (~isequal(dosmooth,'on')) & (~isequal(dosmooth,'off'))
    error(message('stats:lhsdesign:BadSmooth'));
end

% Start with a plain lhs sample over a grid
X = getsample(n,p,dosmooth);

% Create designs, save best one
if isequal(crit,'none') || size(X,1)<2
    maxiter = 0;
end
switch(crit)
    case 'maximin'
        bestscore = score(X,crit);
        for j=2:maxiter
            x = getsample(n,p,dosmooth);
            
            newscore = score(x,crit);
            if newscore > bestscore
                X = x;
                bestscore = newscore;
            end
        end
    case 'correlation'
        bestscore = score(X,crit);
        for iter=2:maxiter
            % Forward ranked Gram-Schmidt step:
            for j=2:p
                for k=1:j-1
                    z = takeout(X(:,j),X(:,k));
                    X(:,k) = (rank(z) - 0.5) / n;
                end
            end
            % Backward ranked Gram-Schmidt step:
            for j=p-1:-1:1
                for k=p:-1:j+1
                    z = takeout(X(:,j),X(:,k));
                    X(:,k) = (rank(z) - 0.5) / n;
                end
            end
            
            % Check for convergence
            newscore = score(X,crit);
            if newscore <= bestscore
                break;
            else
                bestscore = newscore;
            end
        end
end

% ---------------------
function x = getsample(n,p,dosmooth)
x = rand(n,p);
for i=1:p
    x(:,i) = rank(x(:,i));
end
if isequal(dosmooth,'on')
    x = x - rand(size(x));
else
    x = x - 0.5;
end
x = x / n;

% ---------------------
function s = score(x,crit)
% compute score function, larger is better

if size(x,1)<2
    s = 0;       % score is meaningless with just one point
    return
end

switch(crit)
    case 'correlation'
        % Minimize the sum of between-column squared correlations
        c = corrcoef(x);
        s = -sum(sum(triu(c,1).^2));
        
    case 'maximin'
        % Maximize the minimum point-to-point difference
        [~,dist] = knnsearch(x,x,'k',2);
        s = min(dist(:,2));
        
end

% ------------------------
function z=takeout(x,y)

% Remove from y its projection onto x, ignoring constant terms
xc = x - mean(x);
yc = y - mean(y);
b = (xc-mean(xc))\(yc-mean(yc));
z = y - b*xc;

% -----------------------
function r=rank(x)

% Similar to tiedrank, but no adjustment for ties here
[~, rowidx] = sort(x);
r(rowidx) = 1:length(x);
r = r(:);

X = zeros(p,n);   %初始化采样点
X(1,:) = [1,ceil(10*rand(1))];  %第一个点横坐标为1，纵坐标1到10随机
A = [X(1,2)];     %用来存储存在的点所占用在纵坐标的标数
B = [1,2,3,4,5,6,7,8,9,10];
for i = 2:10
    isA = ismember(B,A);
    C = B(~isA);    %记录在该横坐标处剩余可用纵坐标的可用标号
    [~,n] = size(C);
    X(i,1) = i;
    for j = 1:n     %从可用标号依次计算
        X(i,2) = C(j);
        for k = 1:i-1
            d(k) = sum((X(i,:)-X(k,:)).^2);  %计算该点处与之前的点的距离的平方
        end
        distance(j) = min(d);   %最小的距离就是该点处的特征距离
    end
    [~,index] = max(distance);   %找到特征距离最大处的索引
    X(i,2) = C(index);     %最小距离最大的地方就是该点的纵坐标
    A = [A,C(index)];   %加入占用纵坐标
    clear distance   %清除上一次迭代的变量
    clear c
    clear d
end
X(:,1) = X(:,1)/10-rand(10,1)/10;
X(:,2) = X(:,2)/10-rand(10,1)/10;
scatter(X(:,1),X(:,2),'filled')
grid on
xticks(0:0.1:1)
yticks(0:0.1:1)