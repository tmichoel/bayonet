function x_sample = bayonet_gibbs_sampler(y,A,lambda,mu,tau,num,burnin,step)
% BAYONET_GIBBS_SAMPLER - Gibbs sampling of Bayesian elastic net coefficients
%
% BAYONET_GIBBS_SAMPLER implements a Gibbs sampling algorithm to sample
% effect size vectors from the posterior distribution of the Bayesian
% elastic net. 
%
% CALL:
%   x_sample = bayonet_gibbs_sampler(y,A,lambda,mu,tau,num,burnin,step)
% 
% INPUT:
%   y           repsonse data (nx1 vector)
%   A           predictor data (nxp matrix)
%   lambda      L2 penalty parameter
%   mu          L1 penalty parameter
%   tau         inverse temperature parameter
%   num         number of samples
%   burnin      number of coordinate loops to skip before sampling starts
%   step        number of coordinate loops between successive samples
%
% OUTPUT:
%   x_sample    sampled coefficients (pxnum matrix)
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

% convert input
[C,w,p] = input2cw(y,A,lambda);

% initialize with maximum-likelihood solution (tau=infty), this ensures
% that the sampler starts in a region with some probability density
x = init_glmnet(y,A,lambda,mu);
b = w - C*x + diag(C).*x;

% container for samples
x_sample = zeros(p,num);

% burn in
for k=1:burnin
   [x,b] = coord_cycle(x,b,C,mu,tau);
end
x_sample(:,1) = x;

% next sample after step cycles
for k=2:num
%     if mod(k,5e2)==0
%         disp(k);
%     end
   for m=1:step
       [x,b] = coord_cycle(x,b,C,mu,tau);
   end
   x_sample(:,k) = x;
end

function [x,b] = coord_cycle(x,b,C,mu,tau)
% coord_cycle - take a conditional sample once for each coordinate
p = length(x);
for i=1:p
    xinew = bayonet_sample1d(C(i,i),b(i),mu,tau);
    diff = xinew - x(i);
    b = b - diff*C(:,i);
    b(i) = b(i) + diff*C(i,i);
    x(i) = xinew;
end


