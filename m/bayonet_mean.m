function [xhat,uhat] = bayonet_mean(C,w,mu,tau,x,varargin)
% BAYONET_MEAN - Posterior expected effect sizes for the Bayesian elastic net
%
% BAYONET_MEAN implements a coordinate descent algorithm to compute the
% stationary phase approximation to the posterior expected effect sizes in
% the Bayesian elastic net.
%
% CALL:
%   [xhat,uhat] = bayonet_mean(C,w,mu,tau,x,[delta,maxit])
% 
% INPUT:
%   C       predictor x predictor data (see input2cw)
%   w       predictor x response data (see input2cw)
%   mu      L1 penalty parameter
%   tau     inverse temperature parameter, can be a single or sequence of
%           descending values
%   x       initial effect size vector (for instance maximum-likelihood
%           values, see init_glmnet)
%   delta   [optional] convergence threshold
%   maxit   [optional] maximum number of coordinate loops
%
% OUTPUT:
%   xhat    expected effect sizes
%   uhat    uhat = w - C*xhat
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

% A stringent convergence threshold is set by default, which is necessary
% for computing marginal distributions. A more relaxed value (1e-3,1e-4)
% can be set using the optional input arguments if only expectation values
% are sought. A warning is produced if convergence is not reached within
% the maximum number of loops, and then one or both parameters need to be
% changed.
delta_default = 1e-6; 
maxit_default = 5e2;

% check optional inputs
switch nargin
    case 5
        delta = delta_default;
        maxit = maxit_default;
    case 7
        delta = varargin{1};
        maxit = varargin{2};
    otherwise
        error('Wrong number of input arguments.')
end

% initialize b (this was called "a" in the paper appendix F.1)
b = w - C*x + diag(C).*x;

ntau = length(tau);
if ntau==1
    % if only one tau-value, run coordinate descent at this value
    x = coord_descent_loop(x,b,C,mu,tau,delta,maxit);
    xhat = x;
    uhat = w - C*x;
else
    % run sequence wise, each time using previous tau solution as input for
    % next
    xhat = zeros(length(w),ntau);
    uhat = zeros(size(xhat));
    for k=1:ntau
        [x,b] = coord_descent_loop(x,b,C,mu,tau(k),delta,maxit);
        xhat(:,k) = x;
        uhat(:,k) = w - C*x;
    end
end

function [x,b] = coord_descent_loop (x,b,C,mu,tau,delta,maxit)
% coord_descent_loop - single update loop over all coordinates
p = length(x);
diff = delta*ones(p,1);
numit = 0;
while max(abs(diff)) >= delta && numit < maxit
   for i=1:p
       vinew = coord_update(b(i),C(i,i),mu,tau);
       diff(i) = vinew - x(i);
       b = b - diff(i)*C(:,i);
       b(i) = b(i) + diff(i)*C(i,i);
       x(i) = vinew;
   end
    numit = numit + 1;
end
%disp(['Number of iterations: ', num2str(numit)])
if numit==maxit
    disp(max(abs(diff)));
    warning('bayonet:coord_descent_loop:"maximum number of iterations reached"');
end

function xinew = coord_update(bi,Cii,mu,tau)
% coord_update - update single coordinate
pol = [Cii^2; -2*bi*Cii; bi^2-mu^2-Cii/tau; bi/tau];
rts = roots(pol);
urts = bi - Cii*rts;
xinew = rts(urts>-mu & urts<mu);
if length(xinew)~=1 
    % this really shouldn't occur unless there's some weird finite
    % precision error
    disp(urts);
    disp(mu);
    error('Error: %d roots', length(xinew));
end
