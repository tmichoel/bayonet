function [x,Px] = bayonet_marg_approx(xhat,xml,j,y,A,C,w,lambda,mu,tau,numx,sig)
% BAYONET_MARG_APPROX - Approximate one-parameter posterior probablity distribution for the Bayesian elastic net
%
% BAYONET_MARG_APPROX computes a maximum-likelihood based approximation to
% the marginal posterior effect size distribution of the Bayesian elastic
% net for a single predictor.
%
% CALL:
%   [x,Px] = bayonet_marg_approx(xhat,xml,j,y,A,C,w,lambda,mu,tau,numx,sig)
%
% INPUT:
%   xhat    expected effect sizes (see bayonet_mean)
%   xml    	maximum-likelihood effect sizes (see init_glmnet)
%   j       coordinate for which distribution is sought
%   y       response data (nx1 vector)
%   A       predictor data (nxp matrix)
%   C       predictor x predictor data (see input2cw)
%   w       predictor x response data (see input2cw)
%   lambda  L2 penalty parameter
%   mu      L1 penalty parameter
%   tau     inverse temperature parameter
%   numx    number of points (on either side of expectation value) where
%           distribution has to be evaluated 
%   rng     range (around expectation value) over which distribution has to
%           be evaluated 
%
% OUTPUT:
%   x       points where distribution has been evaluated (1 x 2*numx+1 vector)
%   Px      values of the distribution at x (1 x 2*numx+1 vector)
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

p = size(A,2);
idx = (1:p)~=j;

xup = xhat(j)+sig/numx:sig/numx:xhat(j)+sig;
xdown = xhat(j)-sig:sig/numx:xhat(j)-sig/numx;

x = [xdown, xhat(j), xup];

%Px1 = -tau*(C(j,j)*x.^2 - 2*w(j)*x + 2*mu*abs(x));
Px = zeros(size(x));
Hmin = energy(xml,C,w,mu);

x_new = zeros(size(xml));
for k=1:2*numx+1
    % get ML on idx
    xml_idx = init_glmnet(y-x(k)*A(:,j),A(:,idx),lambda,mu);
    x_new(j) = x(k);
    x_new(idx) = xml_idx;
    Px(k) = exp(-tau*(energy(x_new,C,w,mu)-Hmin));
end

function H = energy(x,C,w,mu)
H = x'*C*x - 2*w'*x + 2*mu*norm(x,1);