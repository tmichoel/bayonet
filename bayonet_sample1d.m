function xnew = bayonet_sample1d(C,w,mu,tau)
% mplus = (b+mu)/C;
% mmin = (b-mu)/C;
% Phiplus = normcdf(-sqrt(2*tau*C)*mplus);
% Phimin = normcdf(sqrt(2*tau*C)*mmin);
%p = 1./(1+exp(4*tau*mu*b)*Phiplus/Phimin);


xplus = sqrt(tau/C)*(w+mu);
xmin = -sqrt(tau/C)*(w-mu);

mplus = (w+mu)/C;
mmin = (w-mu)/C;

p = 1./(1+erfcx(xplus)/erfcx(xmin));

if binornd(1,p)==1
   % sample from positive half-line
   xnew = rtnorm(0,inf,mmin,(2*tau*C)^(-0.5));
else
   % sample from negative half-line
   xnew = rtnorm(-inf,0,mplus,(2*tau*C)^(-0.5));
end

