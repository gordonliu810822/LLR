function [obj, t] = Zscore_lowrank_reg(Zstat, loci, LD, ncp, obj0, mask, opts)

if nargin < 7
    opts = [];
    opts.verbose = 1;
    
    opts.eps = 0.01;
    opts.nlam = 10;
    opts.maxIters = 2000;
    opts.epsStopLogLik = 1e-5;
end

%% Get nsnp, nGWAS, nAnn and check compatibility
[nsnp, nGWAS] = size(Zstat);
lamseq = opts.lamseq;
nlam = length(lamseq);

L = length(unique(loci)); % # of block

blocksize = zeros(L,1); % size of each block
for j = 1:L
    blocksize(j) = length(loci(loci==j));
end

% loglik = zeros(opts.maxIters,1);
% loglik_masked = zeros(opts.maxIters,1);

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
countInnerIter = 0;

obj =cell(nlam,1);
%% all Pr(Z_jk|I_ijk = 1) are calculated compared to the baseline Pr(Z_jk|I_0jk = 1);
tic;
for ilam = 1:nlam
    fprintf('EM starts for %d-th lambda\n',ilam);
    loglik = zeros(opts.maxIters,1);
    loglik(1) = -inf;
    for iter = 2:opts.maxIters
        %% E-Step
        % update C1 posterior prob
        for j = 1:L
            %          C1(j,:) = ( Cpi1(j,:).*( sum(exp(logGaussian(loci==j,:)),1) )/blocksize(j) )./  ...
            %                   ( Cpi1(j,:).*( sum(exp(logGaussian(loci==j,:)),1) )/blocksize(j) + (1-Cpi1(j,:)) );
            C1(j,:) = ( Cpi1(j,:).*sumGaussianDistr(j,:) )./( Cpi1(j,:).*sumGaussianDistr(j,:) + (1-Cpi1(j,:)) );
        end
        
        %M-Step: update Cpi1 using Boosting
        [Cpi1, x, x0, rankX, InnerIter] = logistic_LowRank_debuged(C1,lamseq(ilam),x,x0,mask, opts.innerMaxIters);
        countInnerIter = countInnerIter + InnerIter;
        Cpi1(Cpi1>0.999) = 0.999;
        Cpi1(Cpi1<0.001) = 0.001;
        
        % track penalized in-complete loglik
        
        llmat =  Cpi1.* sumGaussianDistr + (1-Cpi1);
        loglik(iter) = sum(sum(log(llmat).*mask)) - lamseq(ilam)*rankX;
        % loglik_masked(iter) = 
        
        if opts.verbose >=1
            if mod(iter,10)==0
                fprintf('%d-th iter: loglik %f\n', iter, loglik(iter));
            end
        end
        
        if (loglik(iter) < loglik(iter-1))
            warning('Info: LogLik deseases! \n');
        else
            % stop criterion
            if (loglik(iter) - loglik(iter-1) < opts.epsStopLogLik)
                %betaAlphaMat((iter+1):end,:) = [];
                loglik(iter+1:end) = [];
                fprintf('Info: Algorithm converges in %d iters because there is no improvements in Log likelihood.\n', iter);
                break;
                
            end
        end
    end

    % posterior prob of I is just E(CI|Z) in memo
    for j = 1:L
        I(loci==j,:) = (exp(logGaussian(loci==j,:))./repmat(sum(exp(logGaussian(loci==j,:)),1),blocksize(j),1)) ...
            .* (ones(blocksize(j),1)*C1(j,:));
    end
    
    obj{ilam}.loglik = loglik;
    obj{ilam}.loglik_masked = sum(sum(log(llmat).*(1 - mask)));
    obj{ilam}.Dall = Dall;
    obj{ilam}.I = I;
    obj{ilam}.C1 = C1;
    obj{ilam}.x = x;
    obj{ilam}.x0 = x0;
end
t = toc;