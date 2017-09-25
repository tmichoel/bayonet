function xhat = bayonet_mean1d(C,w,mu,tau)
% mplus = (b+mu)/C;
% mmin = (b-mu)/C;
% Phiplus = normcdf(-sqrt(2*tau*C)*mplus);
% Phimin = normcdf(sqrt(2*tau*C)*mmin);
%p = 1./(1+exp(4*tau*mu*b)*Phiplus/Phimin);


xplus = sqrt(tau/C)*(w+mu);
xmin = -sqrt(tau/C)*(w-mu);

p = 1./(1+erfcx(xplus)/erfcx(xmin));

xhat = w/C + (1-2*p)*mu/C;
