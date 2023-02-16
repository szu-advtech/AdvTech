clc;
clear;

load ORL_400n_1024d_40c.mat

gnd = y;

cluster_n = length(unique(gnd));          %类数    
Circ = 1; 
objPCA = zeros(5,1);
for ProK = 10:10:50

     [XPCA,mapping] = pca(X,ProK);  
     if isreal(XPCA) == false
            continue;
     end
     [DPCA,CenterPCA] = kmeans(XPCA,cluster_n);

     grpsPCA = DPCA(1:length(gnd));
     ACCindexPCATemp= ACC2(gnd, grpsPCA, cluster_n);
             
     objPCA(Circ) = [ACCindexPCATemp]; 
     fprintf('PCA+kmeeans Clustering: Circ = %d, Prok = %d, ACC = %f\n',Circ, ProK, ACCindexPCATemp);

     Circ = Circ+1;

end

m = floor(0.8*size(X,1));    % 设置1/5为mild noise
Circ = 1; 
objTRPCA = zeros(5,1);
for ProK = 10:10:50

     [iter,re,F,A,b] = trpca(X',ProK,m);
     TRPCAY = (X-ones(size(X,1),1)*b')*F*F';
     if isreal(TRPCAY) == false
            continue;
     end
     [DTRPCA,CenterTRPCA] = kmeans(TRPCAY,cluster_n);

     grpsTRPCA = DTRPCA(1:length(gnd));
     ACCindexTRPCATemp= ACC2(gnd, grpsTRPCA, cluster_n);
             
     objTRPCA(Circ) = ACCindexTRPCATemp; 
     fprintf('TRPCA+kmeeans Clustering: Circ = %d, Prok = %d, ACC = %f\n',Circ, ProK, ACCindexTRPCATemp);

     Circ = Circ+1;

end

x = 10:10:50;
plot(x,objTRPCA,'-*b',x,objPCA,'-or',x,objRPCA,'-.p');
legend('T-RPCA','PCA','RPCA','location','northwest');
xlabel('Dimensions');
ylabel('Accuraacy');


