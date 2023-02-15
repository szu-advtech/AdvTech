function [gbestxf, gbestf, gbest] = CSO_LRA(Datal,BU,BD,npl,iterl,P, nl,meanX)
phi = 1;
c = size(Datal,2)-1;
p = Datal;
p = sortrows(p,c+1);
nl = length(Datal);
p1 = Datal;

p1 = unique(p1,'rows');
x = p1(:,1:c);
y = p1(:,c+1);

% kernal='cubic';
kernal='linear';

L = x; LY =y;
global coefC
coefC = rbfcreate(L',LY','RBFFunction', kernal);
nameR = @YCRBF;
[Nap,fap]=YPSO(@YCRBF,[BD',BU']); %Nap is the best for the model



gbest = [Nap,fap];
gbestf = fap;
gbestxf =feval(nameR,P(:,1:c));


end