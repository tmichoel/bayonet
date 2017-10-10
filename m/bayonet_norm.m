function [logZ,detfun] = bayonet_norm(varargin)
% BAYONET_NORM - Log-partition function for the Bayesian elastic net
%
% BAYONET_NORM computes the stationary phase approximation to the partition
% function of the Bayesian elastic net, expressed on log-scale and relative 
% to the number of predictors. It can be called either using the original
% data (y,A) or converted data (C,w) (cf. input2cw). The former is
% recommended when the number of predictors is large, as it allows to
% compute a p-dimensional determinant as an n-dimensional determinant.
%
% CALL:
%   [logZ,detfun] = bayonet_norm(C,w,mu,tau,xhat,uhat);
%   [logZ,detfun] = bayonet_norm(y,A,lambda,mu,tau,xhat,uhat);
%
% INPUT [converted data]:
%   C       predictor x predictor data (see input2cw)
%   w       predictor x response data (see input2cw)
%
% INPUT [original data]:
%   y       response data (nx1 vector)
%   A       predictor data (nxp matrix)
%   lambda  L2 penalty parameter   
%
% INPUT [common]:
%   mu      L1 penalty parameter
%   tau     inverse temperature parameter
%   xhat    expected effect sizes
%   uhat    uhat = w - C*xhat
%
% OUTPUT:
%   logZ    log(Z)/p
%   detfun  log(determinant term)/p, can be of use for diagnostic purposes
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

switch nargin
    case 6
        C = varargin{1}; 
        w = varargin{2};
        mu = varargin{3};
        tau = varargin{4};
        xhat = varargin{5};
        uhat = varargin{6};
        p = length(w);
        D = diag((tau*(mu^2-uhat.^2).^2)./(mu^2+uhat.^2));
        detfun = 0.5*log(det(C+D))/p;
    case 7
        y = varargin{1};
        A = varargin{2};
        lambda = varargin{3};
        mu = varargin{4};
        tau = varargin{5};
        xhat = varargin{6};
        uhat = varargin{7};
        n = size(A,1);
        p = size(A,2);
        if length(y)==n
            w = 0.5*A'*y/n;
        else
            w = y;
        end
        Dv = (tau*(mu^2-uhat.^2).^2)./(mu^2+uhat.^2)+lambda;
        if p>n
            B = eye(n)+(0.5/n)*A*diag(1./Dv)*A';
            detfun = 0.5*sum(log(Dv))/p + 0.5*sum(log(eig(B)))/p;
        else
            B = (0.5/n)*(A'*A)+diag(Dv);
            detfun = 0.5*sum(log(eig(B)))/p;
        end
end

logZ = tau*(w-uhat)'*xhat/p + log(mu) - 0.5*log(tau) ...
    - 0.5*sum(log(mu^2+uhat.^2))/p - detfun;