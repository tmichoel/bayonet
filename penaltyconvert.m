function [lam,alpha] = penaltyconvert(lambda,mu)
% PENALTYCONVERT - Convert my elastic net penalty parameters to glmnet format
lam = 2*(lambda+mu);
alpha = mu/(lambda+mu);