
% 使用该函数前，应使用了readAllResMat_AndCompare_saveXls，生成了xls文件

% bar(test.*100);
% 画图对比，每一行表示要对比的数据
% 例如：test=[0.7,0.8;
%             0.6,0.5；
%             0.3,0.2];

% 使用方法：
% 直接用 readAllResAndCompare 生成的xls，从里面复制（某一列）内容,然后复制给a  a=[复制内容]
% 例如复制第一列ACC的数据给a，那么生成的就是2个图，一个是NCvsPD的acc对比  一个是NCvsSWEDD的acc对比


methodNum=3;  % 我们选择的方法有 MRI-N,M3T,Proposed
feaSlectNum=5;  % 我们选择的5中特征组合，进行实验。
groupNum=2;     % NCvsPD NCvsSWEED

selectColumn=a;
test=reshape(selectColumn,methodNum,groupNum*feaSlectNum);
test=test';

part1=test(1:feaSlectNum,:);        % NCvsPD
part2=test(feaSlectNum+1:end,:);    % NCvsSWEED


%  ----------------------- NC vs. PD -----------------------------
figure
b=bar(part1.*100);
grid off;
legend('MRI-N','M3T','Proposed','Location','NorthWest');

yRange=get(gca,'Ylim'); % 获取坐标轴上最大、最小刻度的(显示范围)
min=yRange(1);
max=yRange(2);

yMax=1.05*max;
set(gca,'YLim',[min,yMax]);	% 设置y轴的显示范围
set(gca,'ytick',min:5:yMax);	% 设置y轴显示步长 step 间隔，超过Ylim的看不到
set(gca,'YGrid','on');  
ylabel('%');

set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'}); % 对x轴tick设置标签(重命名)
% set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'},'FontSize',20); % 对x轴tick设置标签(重命名)



%  ----------------------- NC vs. SWEDD -----------------------------
figure
b=bar(part2.*100);
grid off;
legend('MRI-N','M3T','Proposed','Location','NorthWest');

yRange2=get(gca,'Ylim'); % 获取坐标轴上最大、最小刻度的(显示范围)
min2=yRange2(1);
max2=yRange2(2);

yMax2=1.05*max2;
set(gca,'YLim',[min,yMax2]);	% 设置y轴的显示范围
set(gca,'ytick',min:5:yMax2);	% 设置y轴显示步长 step 间隔，超过Ylim的看不到
set(gca,'YGrid','on');  
ylabel('%');

set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'}); % 对x轴tick设置标签(重命名)
% set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'},'FontSize',20); % 对x轴tick设置标签(重命名)




% 其他
% get(gca,'xlim');    % 获取坐标轴上最大、最小刻度的(显示范围)
% get(gca,'xtick');   % 获取所有在坐标轴上显示的刻度(可能超过显示的范围)
% set(gca,'tickdir','out');             % 可以让坐标刻度的指向改变 out表示指向外
% set(gca,'ticklength',[0.05 0.025]);   % 设置刻度线的长度
% set(gca,'xminortick','on');           % 显示坐标轴的小刻度尺标
% set(gca,'XGrid','on');                % 打开x轴的网格线
% 我们可以通过获得坐标刻度来对其重命名，注意.4f后的|不能少，少了不能对齐
% set(gca,'xticklabel',sprintf('%.4f|',get(gca,'xtick')));  


