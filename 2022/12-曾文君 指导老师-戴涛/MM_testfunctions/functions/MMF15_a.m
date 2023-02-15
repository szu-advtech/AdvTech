
function y=MMF15_a(x,M,num_of_peak)
% 0<=xi<=1
% Input
%        x             popsize*(M-1+k)
%        M             number of objectives;they are the same
%        num_of_peak   number of peaks/global PSs
% Output 
%        y   popsize*M
% The number of variables and objectives are scalable.
% One global PS/PF, num_of_peak-1 different local PS/PF
% PS:xM=1/(2*np)+(i-1)/(np),i=2,3,...np
% PF: (f_1)^2+(f_2)^2+...+(f_M)^2=(1+g_i^*)^2
if nargin<2
    M=3;
end
if nargin<3
    num_of_peak=2;
end

% if size(x,1)~=1
%     x=reshape(x,1,length(x));
% end
%  
 t=(-0.5*sin(pi*x(:,end-1))+(x(:,end)));
 g=2-exp(-2*log10(2).*((t+1/(2*num_of_peak)-0.1)/0.8).^2).*(sin(num_of_peak*pi.*(t+1/(2*num_of_peak)))).^2;

 y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];

end
%Reference Niching Without Niching Parameters: Particle Swarm Optimization Using a Ring Topology
%       Scalable Test Problems for Evolutionary Multiobjective Optimization
%       Change from DTLZ2





