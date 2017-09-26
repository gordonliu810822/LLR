function [obj] = Zscore_lowrank(Zstat, loci, LD, ncp, obj0, mask, opts)

if nargin < 7
    opts = [];
    opts.verbose = 1;
    
    opts.eps = 0.01;

    opts.maxIters = 2000;
    opts.epsStopLogLik = 1e-5;
end

%% Get nsnp, nGWAS, nAnn and check compatibility
[nsnp, nGWAS] = size(Zstat);

L = length(unique(loci)); % # of block

blocksize = zeros(L,1); % size of each block
for j = 1:L
    blocksize(j) = length(loci(loci==j));
end

loglik = zeros(opts.maxIters,1);
loglik_masked = zeros(opts.maxIters,1);

I = zeros(nsnp, nGWAS); % posterior prob of I
C1 = zeros(L,nGWAS); % posterior prob of C = 1
% Cpi1 = zeros(L,nGWAS); % prior prob of C = 1

Q = zeros(nsnp,nGWAS);
Q0 = zeros(L,nGWAS);

%initialization of Q,Q0;
for j = 1:L
    Q(loci==j) = 1/(blocksize(j) +1);
    Q0(j) = 1/(blocksize(j) +1);
end

%% main EM iterations
logGaussian = zeros(nsnp,nGWAS);
logGaussian0 = zeros(L,nGWAS);
% ZstatTrans = zeros(nsnp,nGWAS);
indexWithinBlock = zeros(nsnp,1);
Dall = zeros(nGWAS,opts.maxIters); % LowRank part
x = zeros(L,nGWAS);
[x0, ~] = GetInit0(obj0.C1);

% offset = log(1./(blocksize+1));
% f = x + repmat(offset,1,nGWAS)+ones(L,1)*x0;
f = x +ones(L,1)*x0;

Cpi1 = 1./(1+exp(-f));
% Cpi1 = prob;

D = cell(L,1);
V = cell(L,1);
for j = 1:L

    [V{j},D{j}] = eig(LD{j});
    indexWithinBlock(loci==j) = 1:blocksize(j);
end

for j = 1:L
    tmp = diag((1./diag(D{j})).^0.5)*(V{j}'*Zstat(loci==j,:));
    logGaussian0(j,:) = -sum(tmp.*tmp,1)/2;
end

cumsumblock = cumsum(blocksize)-blocksize;
for j = 1:L
    Zstatj = Zstat(loci==j,:);
    if opts.ncp == 1
        tmp = Zstatj;
        % tmp = V{j}*diag(1./diag(D{j}))*V{j}'*Zstatj;
        tmp(abs(tmp) < 3.7 ) = 3.7*sign(tmp(abs(tmp) < 3.7));
    end
    if opts.ncp == 0 
        tmp = ncp * ones(blocksize(j),nGWAS);
    end
    
    for i = 1: blocksize(j)
        mutmp = zeros(blocksize(j),nGWAS);
        mutmp(i,:) = tmp(i,:);
        mu = LD{j}*mutmp;
        tmp2 = diag((1./diag(D{j})).^0.5)*(V{j}'*(Zstatj-mu));
        % logGaussian(cumsumblock(j)+i,:) =  -diag((Zstatj-mu)'*V{j}*diag(1./diag(D{j}))*V{j}'*(Zstatj-mu))'/2 - logGaussian0(j,:);
        logGaussian(cumsumblock(j)+i,:) = -sum(tmp2.*tmp2,1)/2 - logGaussian0(j,:);
    end  
end

sumGaussianDistr = zeros(L,nGWAS);
for j = 1:L
    sumGaussianDistr(j,:) = sum(exp(logGaussian(loci==j,:)),1)/blocksize(j);
end
%% all Pr(Z_jk|I_ijk = 1) are calculated compared to the baseline Pr(Z_jk|I_0jk = 1);
tic;
for iter = 1:opts.maxIters
     %% E-Step
     % update C1 posterior prob
     for j = 1:L
%          C1(j,:) = ( Cpi1(j,:).*( sum(exp(logGaussian(loci==j,:)),1) )/blocksize(j) )./  ...
%                   ( Cpi1(j,:).*( sum(exp(logGaussian(loci==j,:)),1) )/blocksize(j) + (1-Cpi1(j,:)) );
         C1(j,:) = ( Cpi1(j,:).*sumGaussianDistr(j,:) )./( Cpi1(j,:).*sumGaussianDistr(j,:) + (1-Cpi1(j,:)) );
     end
     
    %M-Step: update Cpi1 using Boosting   
    grad = (C1-Cpi1).*mask; % working response
    [U,~,V] = svd( grad, 'econ');
    dx = opts.eps*U(:,1)*V(:,1)';
    x = x+ dx;
    [~,D,~] = svd( x, 'econ');
    Dall(:,iter) = diag(D);

    % intercept
    dx0 = opts.eps*sign(sum(grad));
    x0 = x0 + dx0;
    
    f = f + dx + ones(L,1)*dx0;
    Cpi1 = 1./(1+exp(-f));
    
    Cpi1(Cpi1>0.999) = 0.999;
    Cpi1(Cpi1<0.001) = 0.001;

    % track penalized in-complete loglik
    
    llmat =  Cpi1.* sumGaussianDistr + (1-Cpi1);
    loglik(iter) = sum(sum(log(llmat).*mask));
    loglik_masked(iter) = sum(sum(log(llmat).*(1 - mask)));
    
    if opts.verbose >=1
        if mod(iter,10)==0
            fprintf('%d-th iter: loglik %f\n', iter, loglik(iter));
        end
    end
end
% posterior prob of I is just E(CI|Z) in memo
for j = 1:L
    I(loci==j,:) = (exp(logGaussian(loci==j,:))./repmat(sum(exp(logGaussian(loci==j,:)),1),blocksize(j),1)) ...
        .* (ones(blocksize(j),1)*C1(j,:));
end
t = toc;

obj.loglik = loglik;
obj.loglik_masked = loglik_masked;
obj.Dall = Dall;
obj.I = I;
obj.C1 = C1;
obj.x = x;
obj.x0 = x0;
obj.t = t;