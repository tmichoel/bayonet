function logZ = bayonet_norm1d_exact(C,w,mu,tau)
% BAYONET_NORM1D - Exact log-partition function of the Bayesian elastic net if the number of predictors is one
%
% BAYONET_NORM1D computes the exact log-partition function of the Bayesian
% elastic net if the number of predictors is one.  
%
%
% CALL:
%   logZ = bayonet_mean1d(C,w,mu,tau)
%
% INPUT:
%   C       quadratic coefficient (positive number)
%   w       linear coefficiet (number)
%   mu      absolute value coefficient (positive number)
%   tau     inverse temperature
%
% OUTPUT:
%   logZ    log(Z)
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

if abs(w)<mu
    logZ = log(0.5*sqrt(pi/(tau*C))*(erfcx(xplus) + erfcx(xmin)));
else
   if w>=0
      logZ = (tau/C)*(w-mu)^2 - log(2) + 0.5*(log(pi)-log(C*tau)) + ...
          + log(erfc(xmin)) + log(1+erfcx(xplus)/erfcx(xmin));
   else
       logZ = (tau/C)*(w+mu)^2 - log(2) + 0.5*(log(pi)-log(C*tau)) + ...
          + log(erfc(xplus)) + log(1+erfcx(xmin)/erfcx(xplus));
   end
end