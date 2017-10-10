function [x,Px,logZx] = bayonet_marg(xhat,uhat,j,y,A,C,w,lambda,mu,tau,logZ,numx,rng)
% BAYONET_MARG - Marginal one-parameter posterior probablity distribution for the Bayesian elastic net
%
% BAYONET_MARG computes the stationary phase approximation to the marginal
% posterior effect size distribution of the Bayesian elastic net for a
% single predictor.
%
% CALL:
%   [x,Px,logZx] = bayonet_marg(xhat,uhat,j,y,A,C,w,lambda,mu,tau,logZ,numx,sig)
%
% INPUT:
%   xhat    expected effect sizes (see bayonet_mean)
%   uhat    uhat = w - C*xhat (see bayonet_mean)
%   j       coordinate for which distribution is sought
%   y       response data (nx1 vector)
%   A       predictor data (nxp matrix)
%   C       predictor x predictor data (see input2cw)
%   w       predictor x response data (see input2cw)
%   lambda  L2 penalty parameter
%   mu      L1 penalty parameter
%   tau     inverse temperature parameter
%   logZ    log-partition function per coordinate (see bayonet_norm)
%   numx    number of points (on either side of expectation value) where
%           distribution has to be evaluated 
%   rng     range (around expectation value) over which distribution has to
%           be evaluated 
%
% OUTPUT:
%   x       points where distribution has been evaluated (1 x 2*numx+1 vector)
%   Px      values of the distribution at x (1 x 2*numx+1 vector)
%   logZx   log of (p-1)-dimensional integrals in marginal distribution
%           formula, can be of use for diagnostic purposes
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

xup = xhat(j)+rng/numx:rng/numx:xhat(j)+rng;
xdown = xhat(j)-rng:rng/numx:xhat(j)-rng/numx;
x = [xdown, xhat(j), xup];
    

Px1 = -tau*(C(j,j)*x.^2 - 2*w(j)*x + 2*mu*abs(x));
logZx = zeros(size(x));
% at x=xhat(j), we know solution
xb = xhat(idx);
uh = uhat(idx);
logZx(numx+1) = bayonet_norm(y-xhat(j)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);

% go up from x=xhat(j)
for k=numx+2:2*numx+1
    [xb,uh] =  bayonet_mean(C(idx,idx),w(idx)-x(k)*C(idx,j),mu,tau,xb);
    logZx(k) = bayonet_norm(y-x(k)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);
end

% go down from x=xhat(j)
xb = xhat(idx);
for k=numx:-1:1
    [xb,uh] =  bayonet_mean(C(idx,idx),w(idx)-x(k)*C(idx,j),mu,tau,xb);
    logZx(k) = bayonet_norm(y-x(k)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);
end


Px = exp(Px1+p*(((p-1)/p)*logZx-logZ));