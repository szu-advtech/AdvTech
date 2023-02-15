function [f] = f1(W,Theta)
%F1 Summary of this function goes here
%   Detailed explanation goes here


load("generate_channel.mat","K","omiga")
f=0;

for i=1:K

    f=f+omiga(i)*log(1+get_gama(i,W,Theta));

end

