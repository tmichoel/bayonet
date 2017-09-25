function [xhat,uhat] = bayonet_mean(C,w,mu,tau,x,varargin)
% BAYONE_MEAN - Analytic approximation for the BAYesian lassO and elastic NET
%
% [xbay,uhat] = bayonet_mean(C,w,mu,tau,x,varargin)
% 

delta_default = 1e-6;
maxit_default = 1e2;
% check additional inputs
switch nargin
    case 5
        % initialize with maximum-likelihood solution (tau=infty)
        delta = delta_default;
        maxit = maxit_default;
    case 7
        delta = varargin{1};
        maxit = varargin{2};
    otherwise
        error('Wrong number of input arguments.')
end

% initialize b
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
        uhat(:,k) = w - C*x; %wminCx(x,y,A,lambda);
    end
end

function [x,b] = coord_descent_loop (x,b,C,mu,tau,delta,maxit)
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
    warning('bayonet:coord_descent_loop_all:"maximum number of iterations reached"');
end

function xinew = coord_update(bi,Cii,mu,tau)
pol = [Cii^2; -2*bi*Cii; bi^2-mu^2-Cii/tau; bi/tau];
rts = roots(pol);
urts = bi - Cii*rts;
xinew = rts(urts>-mu & urts<mu);
if length(xinew)~=1
    disp(urts);
    disp(mu);
    error('Error: %d roots', length(xinew));
end
