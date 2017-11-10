function [betahat,s2,tval,pval,sigma2] = LinearRegCan2(x,y)


[n,p] = size(x);
if n ~= length(y)
    error('Not the same size of x, y');
end

if p+1 >= n
    error('Ordinary Least Square does not work because p+1 >= n');
end

x = [x, ones(n,1)]; % add intercept
betahat = x\y;

r = y - x*betahat; %residual
sigma2 = r'*r/(n-p-1); % variance of noise (p for variables, 1 for intercept)

xtx = x'*x;
Beta_cov = sigma2*linsolve(xtx,eye(size(xtx)),struct('SYM',true,'POSDEF',true));
tval = betahat./sqrt(diag(Beta_cov));% t-value
pval = 2*tcdf(abs(tval),n-p-1,'upper');% two-side pvalue
s2 = diag(Beta_cov);



