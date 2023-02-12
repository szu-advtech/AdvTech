

% 显示单个数据图，无对比，且在柱形图上方显示对应的值
% 只有一行，表示数据


% t表示一行数据
t=[0.793 0.767 0.781 0.832];
% t=[t;t];
% t=a(1,:);
% bar(t)
len=length(t);
figure;
bar(1:len,t.*100,'b')
grid off;
legend('Test1','Location','NorthEast');

yRange=get(gca,'Ylim'); % 获取坐标轴上最大、最小刻度的(显示范围)
min=yRange(1);
max=yRange(2);

yMax=1.08*max;
set(gca,'YLim',[min,yMax]);	% 设置y轴的显示范围
set(gca,'ytick',min:20:yMax);	% 设置y轴显示步长 step 间隔，超过Ylim的看不到
set(gca,'YGrid','on');  
ylabel('%');

for i=1:len
    y=2+t(i)*100;
    text(i-0.2,y,['\fontsize{15}\color{red}',num2str(t(i)*100),'%']);
end


title('Test2','FontSize',20,'Color','k')
% set(gca,'XColor','red');

set(gca,'XTickLabel',{'acc','sen','spec','f-meas ','prec','auc','MMSE-cc','MMSE-rmse'}); % 对x轴tick设置标签(重命名)
set(gca,'FontSize',15); % 设置坐标轴所有标签字体大小
% axis ij   % 设置坐标走原点为左上
% set(gca,'color','red'); % 设置坐标轴背景颜色

% set(gca,'xcolor','red');
