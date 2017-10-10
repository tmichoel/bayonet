function [x,tau] = init_glmnet(y,A,lambda,mu)
% INIT_GLMNET - Maximum-likelihood initialization for the Bayesian elastic net
%
% INIT_GLMNET finds the maximum-likelihood lasso or elastic net solution
% for given input data and hyper-parameter values. This function converts
% penalty parameters to Glmnet format, and then calls Glmnet, available
% here:
%
% https://web.stanford.edu/~hastie/glmnet_matlab/
%
%
% CALL:
%   [x,tau] = init_glmnet(y,A,lambda,mu)
%
% INPUT:
%   y       response data (nx1 vector)
%   A       predictor data (nxp matrix)
%   lambda  L2 penalty parameter
%   mu      L1 penalty parameter
%
% OUTPUT:
%   x       maximum-likelihood coefficient estimates (px1 vector)
%   tau     maximu-a-posteriori estimate for the inverse temperature
%
%
% This file is part of the Bayonet toolbox for Matlab. Please cite the
% following paper if you use this software:
%
% Michoel T. Analytic solution and stationary phase approximation for the
% Bayesian lasso and elastic net. arXiv:1709.08535 (2017).
%
% https://arxiv.org/abs/1709.08535
%
[lam,alpha] = penaltyconvert(lambda,mu);
opts.alpha = alpha;
opts.lambda = lam;
opts.nlambda = 1;
fit = glmnet(A,y,'gaussian',opts);
x = fit.beta;
n = length(y);
p = size(A,2);
tau = (0.5*n+p)/(0.5*norm(y-A*x)^2/n + lambda*norm(x)^2 + 2*mu*norm(x,1));