clear ;
rmax=20;
dimv=[10,30,50,100];
ub=[5.12,2.048,32.768,600,5];


for dimi=1:4
    
    dim=dimv(dimi);
    for func=1:5
        
        finaly=[];
        c=dim;
        bd=ones(1,c)*-ub(func);
        bu=ones(1,c)*ub(func);
        
        for r=1:rmax
%             offsize=5*c;
%             offdata=lhsdesign(offsize,c).*(ones(offsize,1)*(bu-bd))+ones(offsize,1)*bd;
%             offy=bench_func(offdata,func);
%             L=[offdata,offy];
%             FE=offsize;


% -----------execute GA_SBX until pop reach 6d------------
            pc=1;%Crossover Probability
            pm=1/c;%Mutation Probability
            n=50;%Population Size
            POP = initialize_pop(n,c,bu,bd);
            Y=bench_func(POP,func);
            L=[POP,Y];
            FE=size(L,1);

            while FE<6*c
          
                
                NPOP1=SBX(L,bu,bd,pc,n);
                [ Y ] = bench_func(NPOP1,func);
                FE=FE+size(NPOP1,1);
                NPOP1=[NPOP1,Y];
                NPOP2=mutation(L,bu,bd,pm,n);
                [ Y ] = bench_func(NPOP2,func);
                FE=FE+size(NPOP2,1);
                NPOP2=[NPOP2,Y];
                L=[L;NPOP1;NPOP2];

                YAVE=mean(L(:,c+1:end),2);
                [A,Is]=sort(YAVE);
                POP=[L(Is(1:n),1:c)];
                L=[L(Is(1:n),1:c+1)];

            
                  
            end


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
