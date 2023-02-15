

% ��ʾ��������ͼ���޶Աȣ���������ͼ�Ϸ���ʾ��Ӧ��ֵ
% ֻ��һ�У���ʾ����


% t��ʾһ������
t=[0.793 0.767 0.781 0.832];
% t=[t;t];
% t=a(1,:);
% bar(t)
len=length(t);
figure;
bar(1:len,t.*100,'b')
grid off;
legend('Test1','Location','NorthEast');

yRange=get(gca,'Ylim'); % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
min=yRange(1);
max=yRange(2);

yMax=1.08*max;
set(gca,'YLim',[min,yMax]);	% ����y�����ʾ��Χ
set(gca,'ytick',min:20:yMax);	% ����y����ʾ���� step ���������Ylim�Ŀ�����
set(gca,'YGrid','on');  
ylabel('%');

for i=1:len
    y=2+t(i)*100;
    text(i-0.2,y,['\fontsize{15}\color{red}',num2str(t(i)*100),'%']);
end


title('Test2','FontSize',20,'Color','k')
% set(gca,'XColor','red');

set(gca,'XTickLabel',{'acc','sen','spec','f-meas ','prec','auc','MMSE-cc','MMSE-rmse'}); % ��x��tick���ñ�ǩ(������)
set(gca,'FontSize',15); % �������������б�ǩ�����С
% axis ij   % ����������ԭ��Ϊ����
% set(gca,'color','red'); % ���������ᱳ����ɫ

% set(gca,'xcolor','red');
