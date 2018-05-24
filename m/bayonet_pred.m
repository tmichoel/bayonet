function [z,Pz] = bayonet_pred(xhat,B,C,w,mu,tau,logZ,n,numz,rng)
% BAYONET_PRED - Posterior predictive probablity distribution for the Bayesian elastic net
%
% BAYONET_PRED computes the stationary phase approximation to the posterior
% predictive distribution for the unseen response to a new sample of
% predictor data given existing training data.
%
% CALL:
%   [z,Pz] = bayonet_pred(xhat,B,C,w,lambda,mu,tau,logZ,numx,rng)
%
% INPUT:
%   xhat    expected effect sizes (see bayonet_mean)
%   B       predictor data for m new samples (size m x p) 
%   C       predictor x predictor training data (see input2cw)
%   w       predictor x response training data (see input2cw)
%   lambda  L2 penalty parameter
%   mu      L1 penalty parameter
%   tau     inverse temperature parameter
%   logZ    log-partition function for the training data (see bayonet_norm)
%   n       number of training samples
%   numx    number of points (on either side of expectation value) where
%           distribution has to be evaluated 
%   rng     range (around expectation value) over which distribution has to
%           be evaluated 
%
% OUTPUT:
%   z       points where distribution has been evaluated (m x 2*numx+1 vector)
%   Pz      values of the distribution at x (1 x 2*numx+1 vector)
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

p = size(C,1); % number of predictors
m = size(B,1); % number of new samples
Cnew = C + 0.5*(B'*B)/n; % augmented C matrix
zhat = B*xhat; % expected response values

zup = zhat+rng/numz:rng/numz:zhat+rng;
zdown = zhat-rng:rng/numz:zhat-rng/numz;
z = [zdown, zhat, zup];

% log of Gaussian prefactor
if m==1
    Pz1 = 0.5*m*(log(tau)-log(2*pi*n)) - 0.5*tau.*z.^2./n;
else
    Pz1 = 0.5*m*(log(tau)-log(2*pi*n)) - 0.5*tau.*sum(z.^2)./n; 
end

logZx = zeros(1,size(z,2)); % log of z-dependent partition function

% first fill value at expectation value
wnew = w+0.5*(B'*z(:,numz+1))/n;
[xb,uh] =  bayonet_mean(Cnew,wnew,mu,tau,xhat);
logZx(numz+1) = bayonet_norm(Cnew,wnew,mu,tau,xb,uh);

% go up from expectation value
for k=numz+2:2*numz+1
    wnew = w+0.5*B'*z(:,k)/n;
    [xb,uh] =  bayonet_mean(Cnew,wnew,mu,tau,xb);
    logZx(k) = bayonet_norm(Cnew,wnew,mu,tau,xb,uh);
end

% go down from expectation value
xb = xhat;
for k=numz:-1:1
    wnew = w+0.5*B'*z(:,k)/n;
    [xb,uh] =  bayonet_mean(Cnew,wnew,mu,tau,xb);
    logZx(k) = bayonet_norm(Cnew,wnew,mu,tau,xb,uh);
end

% make posterior probabilities
Pz = exp(Pz1 + p*(logZx - logZ));