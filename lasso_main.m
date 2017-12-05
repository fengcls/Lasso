clear;clc;
%% generate random synthetic data
n = 1000; p = 1000; sparse_ratio = 0.01;
A = randn(n,p);
beta = randn(p,1);

beta(randperm(p,round((1-sparse_ratio)*p)))=0;beta0 = randn;
b = beta0 + A*beta + randn(n,1);

lambda = 0.1;

%% straightforward method with cvx
% cvx_begin
% variable x_bi(p,1)
% minimize(0.5*norm(A*x_bi-b,2)^2+lambda*norm(x_bi,1))
% cvx_end

%%
apga_lasso_func(A,b,lambda);

pga_lasso_func(A,b,lambda);

subgrad_func(A,b,lambda);
