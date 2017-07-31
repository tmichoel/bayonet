function [logZ,detfun] = bayonet_norm(varargin)
% BAYONET_NORM - Analytic approximation for the partition function of the Bayesian Elastic Net
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
        w = 0.5*A'*y/n;
        Dv = (tau*(mu^2-uhat.^2).^2)./(mu^2+uhat.^2)+lambda;
        B = eye(n)+(0.5/n)*A*diag(1./Dv)*A';
        detfun = 0.5*sum(log(Dv))/p + 0.5*sum(log(eig(B)))/p;% 0.5*log(det(eye(n)+(0.5/n)*A*diag(1./Dv)*A'))/p;
end

logZ = tau*(w-uhat)'*xhat/p + log(mu) - 0.5*log(tau) ...
    - 0.5*sum(log(mu^2+uhat.^2))/p - detfun;