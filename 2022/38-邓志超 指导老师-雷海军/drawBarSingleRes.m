



Res=Res_2;
clear cc rmse test
for i=1:length(Res)-1
    cc(i)=Res{i}.ScoreCc;
    rmse(i)=Res{i}.ScoreRmse;
end

cc=[cc,Res{length(Res)}.ScoreCc];


rmse=[rmse,0];

% 每列表示需要对比的数据
test=[cc;rmse];
test=[test(1,:).*100;test(2,:).*10];

figure
b=bar(test);
grid off;
legend('SVM-线性核','SVM-多项式核','SVM-RBF核','SVM-sigmoid核','Location','NorthEast');

yRange=get(gca,'Ylim'); % 获取坐标轴上最大、最小刻度的(显示范围)
min=yRange(1);
max=yRange(2);

yMax=1.05*max;
set(gca,'YLim',[min,yMax]);	% 设置y轴的显示范围
set(gca,'ytick',min:3:yMax);	% 设置y轴显示步长 step 间隔，超过Ylim的看不到
set(gca,'YGrid','on');  
ylabel('%');

set(gca,'XTickLabel',{'CC','RMSE'},'FontSize',20); % 对x轴tick设置标签(重命名)


% 其他
% get(gca,'xlim');    % 获取坐标轴上最大、最小刻度的(显示范围)
% get(gca,'xtick');   % 获取所有在坐标轴上显示的刻度(可能超过显示的范围)
% set(gca,'tickdir','out');             % 可以让坐标刻度的指向改变 out表示指向外
% set(gca,'ticklength',[0.05 0.025]);   % 设置刻度线的长度
% set(gca,'xminortick','on');           % 显示坐标轴的小刻度尺标
% set(gca,'XGrid','on');                % 打开x轴的网格线
% 我们可以通过获得坐标刻度来对其重命名，注意.4f后的|不能少，少了不能对齐
% set(gca,'xticklabel',sprintf('%.4f|',get(gca,'xtick')));  


