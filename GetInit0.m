function [w0, lam, prob] = GetInit0(y)

[n,p] = size(y);
w0_old = zeros(1,p);
f = ones(n,1)*w0_old;
prob = 1./(1+exp(-f));
maxIter = 100;
for iter = 1:maxIter
    zz = (y-prob);
    
    %z =  w_old(2:end,:)+ones(n,1)*w_old(1,:) + zz;
    z = (ones(n,1)*w0_old) + zz/0.25;
    
    w0 = mean(z);
    
    f =  ones(n,1)*w0;
    prob = 1./(1+exp(-f));
    prob(prob>0.999) = 0.999;
    prob(prob<0.001) = 0.001;
   
    gap0 = max(abs(w0-w0_old));
    
    if gap0<1e-4;
        break;
    end
   
    w0_old = w0;
    
end
    
zz = (z-ones(n,1)*w0);
[~,D,~] = svd( zz, 'econ');
lam = D(1,1)/4;