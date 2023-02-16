function [ offy ] = bench_func( offdata,func )
%BENCH_FUNC �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    [n,c]=size( offdata);
    index=ones(n,1);
    offdata=offdata;
    switch(func)
        case 1
            %Ellipsoid Problem
            para=repmat((1:c),n,1);
            result=offdata.^2.*para;
            offy=sum(result,2);
        case 2
            %Rosenbrock Problem
            cutleft=offdata(1:n,2:c);
            cutright=offdata(1:n,1:c-1);
            result=100*((cutleft-cutright.^2).^2)+(cutright-1).^2;
            offy=sum(result,2);
        case 3
            %Ackley Problem
            result=-20*exp(-0.2*sqrt((1/c)*(sum(offdata.^2,2))))-exp((1/c)*sum(cos(2*pi.*offdata),2))+exp(1)+20; 
            offy=result.*index;
        case 4
            %Griewank
            y1 = 1 / 4000 * sum(offdata.^2,2);
            y2 = index ;
            for  h = 1 :c
                  y2 = y2.* cos(offdata(:,h) / sqrt(h));
            end
             offy= y1.*index - y2 + 1 ;
            
        case 5
            %Rastrigin
            offy=sum((offdata.^2 - 10 * cos( 2 * pi * offdata) + 10 ),2).^index;
    end

end

