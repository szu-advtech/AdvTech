



Res=Res_2;
clear cc rmse test
for i=1:length(Res)-1
    cc(i)=Res{i}.ScoreCc;
    rmse(i)=Res{i}.ScoreRmse;
end

cc=[cc,Res{length(Res)}.ScoreCc];


rmse=[rmse,0];

% ÿ�б�ʾ��Ҫ�Աȵ�����
test=[cc;rmse];
test=[test(1,:).*100;test(2,:).*10];

figure
b=bar(test);
grid off;
legend('SVM-���Ժ�','SVM-����ʽ��','SVM-RBF��','SVM-sigmoid��','Location','NorthEast');

yRange=get(gca,'Ylim'); % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
min=yRange(1);
max=yRange(2);

yMax=1.05*max;
set(gca,'YLim',[min,yMax]);	% ����y�����ʾ��Χ
set(gca,'ytick',min:3:yMax);	% ����y����ʾ���� step ���������Ylim�Ŀ�����
set(gca,'YGrid','on');  
ylabel('%');

set(gca,'XTickLabel',{'CC','RMSE'},'FontSize',20); % ��x��tick���ñ�ǩ(������)


% ����
% get(gca,'xlim');    % ��ȡ�������������С�̶ȵ�(��ʾ��Χ)
% get(gca,'xtick');   % ��ȡ����������������ʾ�Ŀ̶�(���ܳ�����ʾ�ķ�Χ)
% set(gca,'tickdir','out');             % ����������̶ȵ�ָ��ı� out��ʾָ����
% set(gca,'ticklength',[0.05 0.025]);   % ���ÿ̶��ߵĳ���
% set(gca,'xminortick','on');           % ��ʾ�������С�̶ȳ߱�
% set(gca,'XGrid','on');                % ��x���������
% ���ǿ���ͨ���������̶���������������ע��.4f���|�����٣����˲��ܶ���
% set(gca,'xticklabel',sprintf('%.4f|',get(gca,'xtick')));  


