%  获取所有的 预测得分 和 目标得分
R=Res1.maxAcc_Ra;
tempPre=[];
tempTarg=[];
len=length(R);
for i=1:len
    tempPre=[tempPre;R{i}.maxAcc_Ldex];   
    tempTarg=[tempTarg;R{i}.maxAcc_Ltest];
end

X=[tempPre,tempTarg];
% corrplot(X);
corr(tempPre,tempTarg)


figure
hold on;
R=Res1.maxAcc_Ra;
% R=Res1.maxBcc_Rb;


cellfun(@(x){scatter(x.maxAcc_Ldex,x.maxAcc_Ltest,'k')},R);

% cellfun(@(x){scatter(x.Adec,x.Atest)},R1);

predictScore=cellfun(@(x){x.maxAcc_Ldex},R);
targetScore=cellfun(@(x){x.maxAcc_Ltest},R);

[n1,n2]=size(predictScore);% 矩阵形式的长度宽度 

predictScore=reshape(predictScore,n1*n2,1); % 将矩阵形式转换成列形式
targetScore=reshape(targetScore,n1*n2,1);


tempPredict=[];
tempTarget=[];
% 将交叉验证中所有的预测、测试标签(注意预测和测试是一一对应的)都以行向量的形式保存
for i=1:size(predictScore,1)
    tempPredict=[tempPredict,predictScore{i}'];
    tempTarget=[tempTarget,targetScore{i}'];    
end


% scatter(tempPredict,tempTarget);
hold on;
p=polyfit(tempPredict,tempTarget,1);
newY=polyval(p,tempPredict);
plot(tempPredict,newY,'-r');
%
% xlim=get(gca,'xlim');
% ylim=get(gca,'ylim');
% xx=(xlim(2)-xlim(1))/2+xlim(1);
% yy=(ylim(2)-ylim(1))/2++ylim(1);
% text(xx,yy,'\fontsize{16}\color{red}ACC=93.22%')

xlabel('Predict Score');
ylabel('Target Score');


xRange=get(gca,'Xlim'); % 获取坐标轴上最大、最小刻度的(显示范围)
yRange=get(gca,'Ylim'); % 获取坐标轴上最大、最小刻度的(显示范围)

xLength=xRange(2)-xRange(1);%分别获取x,y轴的范围
yLength=yRange(2)-yRange(1);

showPercentageX=0.05; % 设置显示text的坐标
showPercentageY=0.9;

text(xRange(1)+xLength*showPercentageX,...
    yRange(1)+yLength*showPercentageY,['\fontsize{15}\color{red}','CC=',...
    num2str(sprintf('%.4f',Res1.maxAcc))]);



