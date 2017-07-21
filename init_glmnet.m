function [x,tau] = init_glmnet(y,A,lambda,mu)
% INIT_GLMNET - Maximum-likelihood initialization for the Bayesian elastic net
[lam,alpha] = penaltyconvert(lambda,mu);
opts.alpha = alpha;
opts.lambda = lam;
opts.nlambda = 1;
fit = glmnet(A,y,'gaussian',opts);
x = fit.beta;
n = length(y);
p = size(A,2);
tau = (0.5*n+p)/(0.5*norm(y-A*x)^2/n + lambda*norm(x)^2 + 2*mu*norm(x,1));
