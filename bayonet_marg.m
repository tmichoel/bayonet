function [x,Px,logZx] = bayonet_marg(xhat,uhat,j,y,A,C,w,lambda,mu,tau,logZ,numx,sig)
% BAYONET_MARG - Marginal one-parameter posterior probablity distribution for the Bayesian Elastic Net

n = size(A,1);
p = size(A,2);
idx = (1:p)~=j;


xup = xhat(j)+sig/numx:sig/numx:xhat(j)+sig;
xdown = xhat(j)-sig:sig/numx:xhat(j)-sig/numx;
x = [xdown, xhat(j), xup];
    

Px1 = -tau*(C(j,j)*x.^2 - 2*w(j)*x + 2*mu*abs(x));
logZx = zeros(size(x));
% at x=xhat(j), we know solution
xb = xhat(idx);
uh = uhat(idx);
logZx(numx+1) = bayonet_norm(y-xhat(j)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);

% go up from x=xhat(j)
for k=numx+2:2*numx+1
    [xb,uh] =  bayonet_mean(C(idx,idx),w(idx)-x(k)*C(idx,j),mu,tau,xb);
    logZx(k) = bayonet_norm(y-x(k)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);
end

% go down from x=xhat(j)
xb = xhat(idx);
for k=numx:-1:1
    [xb,uh] =  bayonet_mean(C(idx,idx),w(idx)-x(k)*C(idx,j),mu,tau,xb);
    logZx(k) = bayonet_norm(y-x(k)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);
end


Px = exp(Px1+p*(((p-1)/p)*logZx-logZ));