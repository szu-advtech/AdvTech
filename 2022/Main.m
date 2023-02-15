clear ;
rmax=20;
dimv=[10,30,50];% dimv=[10,30,50,100];
ub=[5.12,2.048,32.768,600,5];

for dimi=1:3
    
    dim=dimv(dimi);
    for func=1:5
        
        finaly=[];
        c=dim;
        bd=ones(1,c)*-ub(func);
        bu=ones(1,c)*ub(func);
        
        %用LHS采样11-D的数据%
        for r=1:rmax
            offsize=11*c;
            offdata=lhsdesign(offsize,c).*(ones(offsize,1)*(bu-bd))+ones(offsize,1)*bd;
            
            %offdata=rand(offsize,c).*(ones(offsize,1)*(bu-bd))+ones(offsize,1)*bd;
            offy=bench_func(offdata,func);
            
            
            L=[offdata offy];
            
            [time,P,gbest] =DDEA_PES(c,L,bu,bd);
            
            y=bench_func(P,func);
            finaly=[finaly y];
        end
        meany=sum(finaly)/rmax;
        stdy=std(finaly);
        [meany stdy];
        
        filename=['DDEA-PES_func ',num2str(func),'Dim ',num2str(dim),'bserror.txt'];
        
        fp=fopen(filename,'w');
        fprintf(fp,'%f\n',finaly');
        fprintf(fp,'mean: %f\n',meany);
        fprintf(fp,'std: %f\n',stdy);
        fclose(fp);
    end
end
