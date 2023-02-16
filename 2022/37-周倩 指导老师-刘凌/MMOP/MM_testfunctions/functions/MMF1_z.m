function y = MMF1_z(x)
% 1<=x1<=3    -1<=x2<=1
% x       number_of_point * number_of_decision_var
% y       number_of_point * number_of_objective 

    left_index=find(x(:,1)<2);%
    right_index=find(x(:,1)>=2);%
    y(left_index,1)      = 2-x(left_index,1);
    y(right_index,1)      = x(right_index,1)-2;
    y(left_index,2)=1.0 - sqrt(2-x(left_index,1)) + 2.0*(x(left_index,2)-sin(6*pi*(2-x(left_index,1))+pi)).^2;
    y(right_index,2)=1.0 - sqrt(x(right_index,1)-2) + 2.0*(x(right_index,2)-sin(2*pi*(x(right_index,1)-2)+pi)).^2;
end



%{
%% generate PS 
clear;
num=400;%定义点的个数
PS(:,1)=linspace(1,3,num);
%左半部分 右半部分分别处理
    left_index=find(PS(:,1)<2);%找到小于2的部分
    right_index=find(PS(:,1)>=2);%找到大于2的部分
    PS(left_index,2)=sin(6*pi*abs(PS(left_index,1)-2)+pi);
    PS(right_index,2)=sin(2*pi*abs(PS(right_index,1)-2)+pi);
plot(PS(:,1),PS(:,2),'ro')
% plot(PS(:,1),PS(:,2),'ko','MarkerFaceColor','black','MarkerSize',4)
xlabel {\itx}_1
ylabel {\itx}_2
% set(get(gca,'YLabel'),'Rotation', -pi/2,'Position',[1,0]);%将ylabel顺时针旋转90度
PS1=PS(left_index,:);
PS2= PS(right_index,:);
PS_of_MMF13_data

%% generate PF
PF1= MMF1_z(PS1);
PF2=MMF1_z(PS2);
PF_of_MMF11_data
figure, plot(PF1(:,1),PF1(:,2),'o');
figure, plot(PF2(:,1),PF2(:,2),'o');
%}