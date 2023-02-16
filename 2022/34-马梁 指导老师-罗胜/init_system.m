

M=4;
K=4;
N=50;
P_T=5;



G=randn(N,M)+1i*randn(N,M);

h_rk=randn(N,K)+1i*randn(N,K);
h_dk=randn(M,K)+1i*randn(M,K);
sigma=10^(-14.7);
omiga=2.*ones(K,1)';



eta=0.8;  %η
Theta=sqrt(eta)*diag(rand(N,1)+1i*rand(N,1));

save("generate_channel.mat","G","h_rk","h_dk","N","K","M","sigma","omiga","eta","P_T");








