function [out]=soft_thres(a,lam)
% a is a matrix or vector or scalar
% lam is a scalar threshold
% perform softthresholding 

ind=(abs(a)<lam);
out=a-lam;
out(ind)=0;
