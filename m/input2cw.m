function [C,w,p,dCinv] = input2cw(y,A,lambda)
% INPUT2CW - Convert Bayonet input data
%
% INPUT2CW converts input data into the format used by other Bayonet
% functions.
%
% CALL:
%   [C,w,p,dCinv] = input2cw(y,A,lambda)
%
% INPUT:
%   y       response data (nx1 vector)
%   A       predictor data (nxp matrix)
%   lambda  L2 penalty parameter
%
% OUTPUT:
%   C       C = 0.5*(A'*A)/n + lambda*eye(p) (pxp matrix)
%   w       w = 0.5*A'*y/n (px1 vector)
%   p       number of predictors
%   dCinv   diagonal of C^(-1)
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
n = length(y);
p = size(A,2);
C = 0.5*(A'*A)/n + lambda*eye(p);
w = 0.5*A'*y/n;

% diagonal of C-inverse
if nargout==4
    if p<=n
        dCinv = diag(inv(C));
    else
        CT = eye(n)+0.5*(A*A')/(n*lambda);
        B = CT\A;
        dCinv = 1/lambda - 0.5*diag(A'*B)/(n*lambda^2);
   end
end