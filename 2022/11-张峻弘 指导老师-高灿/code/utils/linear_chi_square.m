function x = linear_chi_square(v, u, rho, acc)
% LINEAR_CHI_SQUARE Solves the a particular quadratically constrained
% linear problem.
%
% x = linear_chi_square(v, u, rho, acc) sets x to be the solution to the
%   optimization problem
%
% min. x' * v
% s.t. norm(x - u, 2)^2 <= rho
%      sum(x) == 1, x >= 0.
%
% Uses a binary search strategy along with projections onto the simplex to
% solve the problem in O(n log (n / acc)) time to solve to accuracy acc (in
% duality gap)


% A partial dual to the problem is given by the Lagrangian
%
% L(x, lambda) = (lambda/2) * (norm(x - u, 2)^2 - rho) + x' * v
% subject to     sum(x) == 1, x >= 0.
%
% Then we maximize lambda over inf_x L(x, lambda), the infimum taken over
% the constrained set.

if (nargin < 4)
  acc = 1e-8;
end

duality_gap = Inf;

max_lambda = Inf;
min_lambda = 0;

x = project_onto_simplex(u, 1);

if (norm(x - u, 2)^2 > rho)
  error('Problem is not feasible');
end

start_lambda = 1;
while (isinf(max_lambda))
  x = project_onto_simplex(u - v / start_lambda, 1);
  lam_grad = .5 * norm(x - u, 2)^2 - rho/2;
  if (lam_grad < 0)
    max_lambda = start_lambda;
  else
    start_lambda = start_lambda * 2;
  end
end

while (max_lambda - min_lambda > acc * start_lambda)
  lambda = (min_lambda + max_lambda) / 2;
  x = project_onto_simplex(u - v / lambda, 1);
  lam_grad = .5 * norm(x - u, 2)^2 - rho/2;
  if (lam_grad < 0)
    % Then lambda is too large, so decrease max_lambda
    max_lambda = lambda;
  else
    min_lambda = lambda;
  end
end
