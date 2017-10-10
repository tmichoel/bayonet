function x_sample = bayonet_sample1d(C,w,mu,tau)
% BAYONET_SAMPLE1D - Exact sampling of Bayesian elastic net coefficients if the number of predictors is one
%
% BAYONET_SAMPLE1D samples a single Bayesian elastic net coefficient from
% the exact posterior distribution if the number of predictors is one.
%
% This function uses the rtnorm function to sample from a truncated normal
% distribution, available here:
%
% http://miv.u-strasbg.fr/mazet/rtnorm/
%
% CALL:
%   x_sample = bayonet_sample1d(C,w,mu,tau)
%
% INPUT:
%   C           quadratic coefficient (positive number)
%   w           linear coefficiet (number)
%   mu          absolute value coefficient (positive number)
%   tau         inverse temperature
%
% OUTPUT:
%   x_sample    new sample (number)
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

xplus = sqrt(tau/C)*(w+mu);
xmin = -sqrt(tau/C)*(w-mu);

mplus = (w+mu)/C;
mmin = (w-mu)/C;

p = 1./(1+erfcx(xplus)/erfcx(xmin));

if binornd(1,p)==1
   % sample from positive half-line
   x_sample = rtnorm(0,inf,mmin,(2*tau*C)^(-0.5));
else
   % sample from negative half-line
   x_sample = rtnorm(-inf,0,mplus,(2*tau*C)^(-0.5));
end

