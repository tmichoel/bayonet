function [x,Px] = bayonet_marg_approx(xhat,xml,j,y,A,C,w,lambda,mu,tau,numx,sig)
% BAYONET_MARG_APPROX - Approximate marginal one-parameter posterior probablity distribution for the Bayesian Elastic Net

n = size(A,1);
p = size(A,2);
idx = (1:p)~=j;

xup = xhat(j)+sig/numx:sig/numx:xhat(j)+sig;
xdown = xhat(j)-sig:sig/numx:xhat(j)-sig/numx;

x = [xdown, xhat(j), xup];

%Px1 = -tau*(C(j,j)*x.^2 - 2*w(j)*x + 2*mu*abs(x));
Px = zeros(size(x));
Hmin = energy(xml,C,w,mu);
% % always use the same solution
% xb = xhat(idx);
% uh = uhat(idx);
% 
% % first get logZx at xhat(j)
% logZx(numx+1) = bayonet_norm(y-xhat(j)*A(:,j),A(:,idx),lambda,mu,tau,xb,uh);
% %what = w(idx) - xhat(j)*C(idx,j);
% 
% % then get logZx at other values
% lzdiff = tau*C(idx,j)'*xb/(p-1); 

% get 
x_new = zeros(size(xml));
for k=1:2*numx+1
    % get ML on idx
    xml_idx = init_glmnet(y-x(k)*A(:,j),A(:,idx),lambda,mu);
    x_new(j) = x(k);
    x_new(idx) = xml_idx;
    Px(k) = exp(-tau*(energy(x_new,C,w,mu)-Hmin));
end



function H = energy(x,C,w,mu)
H = x'*C*x - 2*w'*x + 2*mu*norm(x,1);