function [ POP_new ] = GA_SBX( POP,upperbound,lowerbound,pc,pm,n )
% Usage: [ POP_new ] = GA_SBX( POP,upperbound,lowerbound,pc,pm,n )
%
% Input:
% upperbound            -Upper Bound
% lowerbound            -Lower Bound
% POP           -Input Population
% pc            -Crossover Probability
% pm            -Mutation Probability
% n             -Population Size
%
% Output: 
% POP_new          -The Generated New Population
%------------------------------- Copyright --------------------------------
% Copyright 2020. You are free to use this code for research purposes.All 
% publications which use this code should reference the following papaer:
% Jian-Yu Li, Zhi-Hui Zhan, Chuan Wang, Hu Jin, Jun Zhang, Boosting  
% data-driven evolutionary algorithm with localized data generation, IEEE 
% Transactions on Evolutionary Computation, DOI: 10.1109/TEVC.2020.2979740.
%--------------------------------------------------------------------------

POP_crossover=[];
eta_c=15;
N=size(POP,1);
C=size(upperbound,2);
y=1;
for i=1:n
    r1=rand;
    if r1<=pc
        A=randperm(N);
        k=i;

        if A(2)<A(1)
            y=A(2);
        else
            y=A(1);
        end
        if k==y
            k=A(3);
        end
        d=(sum((POP(y,1:C)-POP(k,1:C)).^2)).^0.5;
        if k~=y
            for j=1:C
                par1=POP(y,j);par2=POP(k,j);
                yd=lowerbound(j);yu=upperbound(j);
                r2=rand;
                if r2<=0.5
                    y1=min(par1,par2);y2=max(par1,par2);
                    if (y1-yd)>(yu-y2)
                        beta=1+2*(yu-y2)/(y2-y1);
                    else
                        beta=1+2*(y1-yd)/(y2-y1);
                    end
                    expp=eta_c+1;beta=1/beta;alpha=2.0-beta^(expp);
                    r3=rand;
                    if r3<=1/alpha
                        alpha=alpha*r3;expp=1/(eta_c+1.0);
                        betaq=alpha^(expp);
                    else
                        alpha=1/(2.0-alpha*r3);expp=1/(eta_c+1);
                        betaq=alpha^(expp);
                    end
                    chld1=0.5*((y1+y2)-betaq*(y2-y1));
                    chld2=0.5*((y1+y2)+betaq*(y2-y1));   
                    aa=max(chld1,yd);
                    bb=max(chld2,yd);
                    if rand>0.5
                        POP_crossover(2*i-1,j)=min(aa,yu);
                        POP_crossover(2*i,j)=min(bb,yu);
                    else
                        POP_crossover(2*i,j)=min(aa,yu);
                        POP_crossover(2*i-1,j)=min(bb,yu);
                    end
                else
                    POP_crossover(2*i-1,j)=par1;
                    POP_crossover(2*i,j)=par2;
                end
            end
        end
    end
    
end

eta_m=15;
POP_new=POP_crossover(:,1:C);
for i=1:n
    k=i;
    POP_new(i,:)=POP_crossover(k,1:C);
    for j=1:C
        r1=rand;
        if r1<=pm
            y=POP_crossover(k,j);
            yd=lowerbound(j);yu=upperbound(j);
            if y>yd
                if (y-yd)<(yu-y)
                    delta=(y-yd)/(yu-yd);
                else
                    delta=(yu-y)/(yu-yd);
                end
                r2=rand;
                indi=1/(eta_m+1);
                if r2<=0.5
                    xy=1-delta;
                    val=2*r2+(1-2*r2)*(xy^(eta_m+1));
                    deltaq=val^indi-1;
                else
                    xy=1-delta;
                    val=2*(1-r2)+2*(r2-0.5)*(xy^(eta_m+1));
                    deltaq=1-val^indi;
                end
                y=y+deltaq*(yu-yd);
                POP_new(i,j)=min(y,yu);POP_new(i,j)=max(y,yd);
            else
                POP_new(i,j)=rand*(yu-yd)+yd;
            end
        end
    end
end
end


