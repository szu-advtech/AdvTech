function obj = Linear(var,~)
    filename = "MFEA/Data/data.mat";
    data = load(filename);
    data = data.data_MFEA.EvBestFitness;%读取目标值
    y1 = data(:,1);
    y2 = data(:,2);
    dim=length(var);
    ny1=y1(1:dim);
    ny2=y2(1:dim);
    p1=polyfit(var,ny1,8);%对数据进行二项式拟合
    p2=polyfit(var,ny2,8);
    p=0.4*p1+0.6*p2;%对拟合的结果进行加权平均
    for i=1:dim
        temp = polyval(p,var(i));%利用拟合的图形计算数据结果
        obj = obj + temp;
    end
end
