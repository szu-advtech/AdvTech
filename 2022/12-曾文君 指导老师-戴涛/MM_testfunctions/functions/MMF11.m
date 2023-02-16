
function y=MMF11(x)
% 0.1<=x1<=1.1    0.1<=x2<=1.1
% 1 Global PS and num_of_peak-1 local PSs
 y=zeros(2,1);
 num_of_peak=2;
 x1=x(1);
 x2=x(2);
 temp1=(sin(num_of_peak*pi.*x2)).^6;
 temp2=exp(-2*log10(2).*((x2-0.1)/0.8).^2);
 g=2-temp2.*temp1;
 y(1)=x1;
 y(2)=g/x1;
end
% Niching Without Niching Parameters: Particle Swarm Optimization Using a Ring Topology
%        Multi-objective Genetic Algorithms: Problem Difficulties and Construction of Test Problems

