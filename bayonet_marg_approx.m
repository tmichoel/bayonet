function [x,Px,logZx] = bayonet_marg_approx(xhat,j,y,A,C,w,lambda,mu,tau,logZ,numx,dCinv,numsig)
% BAYONET_MARG_APPROX - Approximate marginal one-parameter posterior probablity distribution for the Bayesian Elastic Net

n = size(A,1);
p = size(A,2);
idx = (1:p)~=j;

sig = numsig*sqrt(dCinv/(2*tau));
xup = xhat(j)+sig/numx:sig/numx:xhat(j)+sig;
xdown = xhat(j)-sig:sig/numx:xhat(j)-sig/numx;

x = [xdown, xhat(j), xup];

Px1 = -tau*(C(j,j)*x.^2 - 2*w(j)*x + 2*mu*abs(x));
logZx = zeros(size(x));

% always use the same solution
xb = xhat(idx);
uh = uhat(idx);

for k=1:2*numx+1
    logZx(k) = bayonet_norm(y-x(k)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);
end

Px = exp(Px1+p*(((p-1)/p)*logZx-logZ));