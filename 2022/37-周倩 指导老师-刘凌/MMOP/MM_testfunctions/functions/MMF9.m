
function y=MMF9(x)
% 0.1<x1<=1.1    0.1<=x2<=1.1   g(x2)>0
% num_of_peak global PSs
 y=zeros(2,1);
 num_of_peak=2;
 temp1=(sin(num_of_peak*pi.*x(2))).^6;
 g=2-temp1;
 y(1)=x(1);
 y(2)=g/x(1);
end
%Reference Niching Without Niching Parameters: Particle Swarm Optimization Using a Ring Topology
%        Multi-objective Genetic Algorithms: Problem Difficulties and Construction of Test Problems

