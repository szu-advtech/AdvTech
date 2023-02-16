function P = PCA(X, k_PCA)
[~, n] = size(X);
one_vec = ones(n, 1);
H = eye(n) - one_vec'*one_vec / n;
Cov_X = X*H*H'*X';
[V,D] = eigs(Cov_X, k_PCA);
P=V;
end