function xhat = bayonet_mean1d(C,w,mu,tau)
% BAYONET_MEAN1D - Exact posterior expected effect size of the Bayesian elastic net coefficient if the number of predictors is one
%
% BAYONET_MEAN1D computes the exact posterior expected effect size of the
% Bayesian elastic net coefficient if the number of predictors is one. 
%
%
% CALL:
%   xhat = bayonet_mean1d(C,w,mu,tau)
%
% INPUT:
%   C       quadratic coefficient (positive number)
%   w       linear coefficiet (number)
%   mu      absolute value coefficient (positive number)
%   tau     inverse temperature
%
% OUTPUT:
%   xhat    expectation value (number)
%
%
% This file is part of the Bayonet toolbox for Matlab. Please cite the
% following paper if you use this software:
%
% Michoel T. Analytic solution and stationary phase approximation for the
% Bayesian lasso and elastic net. arXiv:1709.08535 (2017).
%
% https://arxiv.org/abs/1709.08535

xplus = sqrt(tau/C)*(w+mu);
xmin = -sqrt(tau/C)*(w-mu);

p = 1./(1+erfcx(xplus)/erfcx(xmin));

xhat = w/C + (1-2*p)*mu/C;
