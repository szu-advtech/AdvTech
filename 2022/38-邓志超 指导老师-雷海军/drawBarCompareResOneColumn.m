
% ʹ�øú���ǰ��Ӧʹ����readAllResMat_AndCompare_saveXls��������xls�ļ�

% bar(test.*100);
% ��ͼ�Աȣ�ÿһ�б�ʾҪ�Աȵ�����
% ���磺test=[0.7,0.8;
%             0.6,0.5��
%             0.3,0.2];

% ʹ�÷�����
% ֱ���� readAllResAndCompare ���ɵ�xls�������渴�ƣ�ĳһ�У�����,Ȼ���Ƹ�a  a=[��������]
% ���縴�Ƶ�һ��ACC�����ݸ�a����ô���ɵľ���2��ͼ��һ����NCvsPD��acc�Ա�  һ����NCvsSWEDD��acc�Ա�


methodNum=3;  % ����ѡ��ķ����� MRI-N,M3T,Proposed
feaSlectNum=5;  % ����ѡ���5��������ϣ�����ʵ�顣
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

yRange=get(gca,'Ylim'); % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
min=yRange(1);
max=yRange(2);

yMax=1.05*max;
set(gca,'YLim',[min,yMax]);	% ����y�����ʾ��Χ
set(gca,'ytick',min:5:yMax);	% ����y����ʾ���� step ���������Ylim�Ŀ�����
set(gca,'YGrid','on');  
ylabel('%');

set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'}); % ��x��tick���ñ�ǩ(������)
% set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'},'FontSize',20); % ��x��tick���ñ�ǩ(������)



%  ----------------------- NC vs. SWEDD -----------------------------
figure
b=bar(part2.*100);
grid off;
legend('MRI-N','M3T','Proposed','Location','NorthWest');

yRange2=get(gca,'Ylim'); % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
min2=yRange2(1);
max2=yRange2(2);

yMax2=1.05*max2;
set(gca,'YLim',[min,yMax2]);	% ����y�����ʾ��Χ
set(gca,'ytick',min:5:yMax2);	% ����y����ʾ���� step ���������Ylim�Ŀ�����
set(gca,'YGrid','on');  
ylabel('%');

set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'}); % ��x��tick���ñ�ǩ(������)
% set(gca,'XTickLabel',{'T1G','T1C','DTI','GCD','GCDS'},'FontSize',20); % ��x��tick���ñ�ǩ(������)




% ����
% get(gca,'xlim');    % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
% get(gca,'xtick');   % ��ȡ����������������ʾ�Ŀ̶�(���ܳ�����ʾ�ķ�Χ)
% set(gca,'tickdir','out');             % ����������̶ȵ�ָ��ı� out��ʾָ����
% set(gca,'ticklength',[0.05 0.025]);   % ���ÿ̶��ߵĳ���
% set(gca,'xminortick','on');           % ��ʾ�������С�̶ȳ߱�
% set(gca,'XGrid','on');                % ��x���������
% ���ǿ���ͨ���������̶���������������ע��.4f���|�����٣����˲��ܶ���
% set(gca,'xticklabel',sprintf('%.4f|',get(gca,'xtick')));  


