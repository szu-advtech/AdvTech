function x = project_onto_simplex(v, B)
% PROJECT_ONTO_SIMPLEX Projects a vector onto a (scaled) probability
% simplex
%
% x = project_onto_simplex(v, B) projects the vector v onto the scaled
%   probability simplex, which is defined by the set
%
%   {x >= 0, sum(x) == B}
%
%   The projection is computed using Euclidean distance.

u = sort(v, 'descend');
sv = cumsum(u);
rho = find(u > (sv - B) ./ (1:length(v))', 1, 'last');
%theta = max(0, (sv(rho) - B) / rho);
theta = (sv(rho) - B) / rho;
x = max(v - theta, 0);
