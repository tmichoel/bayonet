function [lam,alpha] = penaltyconvert(lambda,mu)
% PENALTYCONVERT - Convert Bayonet elastic net penalty parameters to glmnet format
%
% PENALTYCONVERT converts Bayonet's convention for the L1 and L2 penalty
% parameters into Glmnet's convention.
%
% This file is part of the Bayonet toolbox for Matlab. Please cite the
% following paper if you use this software:
%
% Michoel T. Analytic solution and stationary phase approximation for the
% Bayesian lasso and elastic net. arXiv:1709.08535 (2017).
%
% https://arxiv.org/abs/1709.08535
%

lam = 2*(lambda+mu);
alpha = mu/(lambda+mu);