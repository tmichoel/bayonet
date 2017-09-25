function [C,w,p,dCinv] = input2cw(y,A,lambda)
% INPUT2CW - Convert data input for Bayesian elastic net
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