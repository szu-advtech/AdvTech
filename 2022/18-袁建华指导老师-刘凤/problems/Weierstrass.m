function obj = Weierstrass(var,MM)
    %Weierstrass函数，MM为随机旋转矩阵
    kkmax = 20;%kmax设置为20
    dim = length(MM);
    var = var(:,1:dim);
    [NN,dim] = size(var);
    opt=0*ones(NN,dim);
    var = (MM*(var-opt)')'; 
    kmax = repmat(1:kkmax,[NN,1]);
    a = 0.5*ones(NN,kkmax);
    b = 3*ones(NN,kkmax);
    obj = 0;
    for i =1:dim
        obj = obj + sum((a.^kmax).*cos(2*pi*(b.^kmax).*(repmat(var(:,i),[1,kkmax])+0.5)),2);
    end
    obj = obj - sum(dim*(a.^kmax).*cos(2*pi*(b.^kmax)*0.5),2);
end