function obj = Sphere(var,MM)
    %Sphere������MMΪ�����ת����
    dim = length(MM);
    var = var(:,1:dim);
    [NN,dim] = size(var);
    opt=0*ones(NN,dim);
    var = (var - opt);
    obj=sum(var.^2,2);
end