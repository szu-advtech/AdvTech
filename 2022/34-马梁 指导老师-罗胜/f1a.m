function [f] = f1a(W,Theta,alpha)
f=0;
load("generate_channel.mat","omiga","K");
for i=1:K
    f=f+omiga(i)*log(1+alpha(i))-omiga(i)*alpha(i)+omiga(i)*(1+alpha(i))*get_gama(i,W,Theta)/(1+get_gama(i,W,Theta));
end
end
