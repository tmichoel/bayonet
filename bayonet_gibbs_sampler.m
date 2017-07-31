function x_sample = bayonet_gibbs_sampler(y,A,lambda,mu,tau,num,burnin,step)
% BAYONET_GIBBS_SAMPLER - Gibbs sampling of Bayesian elastic net coefficients

n = length(y);
p = size(A,2);

% convert input
[C,w] = input2cw(y,A,lambda);

% initialize with maximum-likelihood solution (tau=infty)
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
   for m=1:step
       [x,b] = coord_cycle(x,b,C,mu,tau);
   end
   x_sample(:,k) = x;
end

function [x,b] = coord_cycle(x,b,C,mu,tau)
p = length(x);
for i=1:p
    xinew = bayonet_sample1d(b(i),C(i,i),mu,tau);
    diff = xinew - x(i);
    b = b - diff*C(:,i);
    b(i) = b(i) + diff*C(i,i);
    x(i) = xinew;
end

% function xinew = coord_update(bi,Cii,mu,tau)
% mplus = (bi+mu)/Cii;
% mmin = (bi-mu)/Cii;
% Phiplus = normcdf(-sqrt(2*tau*Cii)*mplus);
% Phimin = normcdf(sqrt(2*tau*Cii)*mmin);
% p = 1./(1+exp(4*tau*mu*bi)*Phiplus/Phimin);
% if binornd(1,p)==1
%    % sample from positive half-line
%    xinew = rtnorm(0,inf,mmin,(2*tau*Cii)^(-0.5));
% else
%    % sample from negative half-line
%    xinew = rtnorm(-inf,0,mplus,(2*tau*Cii)^(-0.5));
% end


