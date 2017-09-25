function logZ = bayonet_norm1d_exact(C,w,mu,tau)

xplus = sqrt(tau/C)*(w+mu);
xmin = -sqrt(tau/C)*(w-mu);

if abs(w)<mu
    logZ = log(0.5*sqrt(pi/(tau*C))*(erfcx(xplus) + erfcx(xmin)));
else
   if w>=0
      logZ = (tau/C)*(w-mu)^2 - log(2) + 0.5*(log(pi)-log(C*tau)) + ...
          + log(erfc(xmin)) + log(1+erfcx(xplus)/erfcx(xmin));
   else
       logZ = (tau/C)*(w+mu)^2 - log(2) + 0.5*(log(pi)-log(C*tau)) + ...
          + log(erfc(xplus)) + log(1+erfcx(xmin)/erfcx(xplus));
   end
end