function obj = Rastrigin(var,MM)
    %Rastrigin������MMΪ�����ת����
    dim = length(MM);
    var = var(:,1:dim);
    [NN,dim] = size(var);
    opt=0*ones(NN,dim);
    var = (MM*(var-opt)')'; 
    obj = 10*dim + sum(var.^2-10*cos(2*pi*var),2);
end