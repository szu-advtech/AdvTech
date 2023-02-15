clear all;

rand('state',0);
randn('state',0);

%%%%%%%%%%%%%%%%%%%%%% Generating data
dataset=1;

switch dataset
  case 1
   a=0;
   x2scale=1;
  case 2
   a=5;
   x2scale=1;
  case 3
   a=0;
   x2scale=1.7;
end

n1=2;
n2=200;
X1 =[0.1*randn(2,n1/2)+repmat([-3; a],[1 n1/2])];
X2 =[0.1*randn(2,n1/2)+repmat([ 3; 0],[1 n1/2])];
X01=[randn(2,n2/2).*repmat([1;2],[1 n2/2])+repmat([-3; 0],[1 n2/2])];
X02=[randn(2,n2/2).*repmat([1;2],[1 n2/2])+repmat([ 3; 0],[1 n2/2])];
X=[X1 X2 X01 X02];
X(2,:)=X(2,:)*x2scale;
YY=[ones(1,n1/2) 2*ones(1,n1/2) -ones(1,n2/2) -2*ones(1,n2/2)];
Y=YY; Y(YY<0)=0;

%%%%%%%%%%%%%%%%%%%%%% Computing SELF solution
[ T0, Z0]=SELF(X,Y,0.001,1,'weighted',1);
[T09,Z09]=SELF(X,Y,0.9,1,'weighted',1);
[ T1, Z1]=SELF(X,Y,1,1,'weighted',1); 

%%%%%%%%%%%%%%%%%%%%%% Displaying original 2D data
figure(1)
clf
hold on

set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)

h=plot([ -T0(1)  T0(1)]*100,[ -T0(2)  T0(2)]*100,'g--','LineWidth',3);
h=plot([-T09(1) T09(1)]*100,[-T09(2) T09(2)]*100,'m-','LineWidth',3);
h=plot([ -T1(1)  T1(1)]*100,[ -T1(2)  T1(2)]*100,'k:','LineWidth',3);
legend('LFDA','SELF','PCA')
h=plot(X(1,YY== 1),X(2,YY== 1),'bo','MarkerSize',10,'MarkerFaceColor','b');
h=plot(X(1,YY== 2),X(2,YY== 2),'rs','MarkerSize',10,'MarkerFaceColor','r');
h=plot(X(1,YY==-1),X(2,YY==-1),'bo','MarkerSize',5);
h=plot(X(1,YY==-2),X(2,YY==-2),'rs','MarkerSize',5);
axis equal
axis([-8 8 -8 8])
title('Original 2D data and subspace found by SELF')

set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 12]);
print('-depsc',sprintf('SELF%g',dataset))

