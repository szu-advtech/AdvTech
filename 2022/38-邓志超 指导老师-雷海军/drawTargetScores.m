

hold on;
R=Res1.maxBcc_Rb;
meanBcc=Res1.maxBcc;

cellfun(@(x){scatter(x.maxBcc_Ldex,x.maxBcc_Ltest)},R);

% cellfun(@(x){scatter(x.Adec,x.Atest)},R1);

predictScore=cellfun(@(x){x.maxBcc_Ldex},R);
targetScore=cellfun(@(x){x.maxBcc_Ltest},R);

[n1,n2]=size(predictScore);% ������ʽ�ĳ��ȿ�� 

predictScore=reshape(predictScore,n1*n2,1); % ��������ʽת��������ʽ
targetScore=reshape(targetScore,n1*n2,1);


tempPredict=[];
tempTarget=[];
% ��������֤�����е�Ԥ�⡢���Ա�ǩ(ע��Ԥ��Ͳ�����һһ��Ӧ��)��������������ʽ����
for i=1:size(predictScore,1)
    tempPredict=[tempPredict,predictScore{i}'];
    tempTarget=[tempTarget,targetScore{i}'];    
end


% scatter(tempPredict,tempTarget);
hold on;
p=polyfit(tempPredict,tempTarget,1);
newY=polyval(p,tempPredict);
plot(tempPredict,newY,'-r');

% xlim=get(gca,'xlim');
% ylim=get(gca,'ylim');
% xx=(xlim(2)-xlim(1))/2+xlim(1);
% yy=(ylim(2)-ylim(1))/2++ylim(1);
% 
% text(xx,yy,['\fontsize{16}\color{red}meanBcc=',sprintf('%0.3f\n',meanBcc)])

xlabel('Predict Score');
ylabel('Target Score');



xRange=get(gca,'Xlim'); % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
yRange=get(gca,'Ylim'); % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)


xLength=xRange(2)-xRange(1);%�ֱ��ȡx,y��ķ�Χ
yLength=yRange(2)-yRange(1);

showPercentageX=0.05; % ������ʾtext������
showPercentageY=0.9;

text(xRange(1)+xLength*showPercentageX,...
    yRange(1)+yLength*showPercentageY,['\fontsize{15}\color{red}','CC=',...
    num2str(sprintf('%.4f',Res1.maxBcc))]);
