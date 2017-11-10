function obj = Zscore_init(Zstat, loci, LD, ncp, opts)

if nargin < 5
    opts = [];
    opts.verbose = 1;
    
    opts.eps = 0.001;

    opts.maxIters = 1000;
    opts.epsStopLogLik = 1e-5;
    opts.ncp = 1;
end

%% Get nsnp, nGWAS, nAnn and check compatibility
[nsnp, nGWAS] = size(Zstat);

L = length(unique(loci)); % # of block

blocksize = zeros(L,1); % size of each block
for j = 1:L
    blocksize(j) = length(loci(loci==j));
end

loglik = zeros(opts.maxIters,1);
loglik(1) = -inf;

I = zeros(nsnp, nGWAS); % posterior prob of I
C1 = zeros(L,nGWAS); % posterior prob of C = 1
Cpi1 = 1/2*ones(1,nGWAS); % prior prob of C = 1

% Q = zeros(nsnp,nGWAS);
% Q0 = zeros(L,nGWAS);

%initialization of Q,Q0;
% for j = 1:L
%     Q(loci==j) = 1;
%     Q0(j) = 1/(blocksize(j));
% end

%% main EM iterations
logGaussian = zeros(nsnp,nGWAS);
logGaussian0 = zeros(L,nGWAS);
indexWithinBlock = zeros(nsnp,1);

D = cell(L,1);
V = cell(L,1);
for j = 1:L

    [V{j},D{j}] = eig(LD{j});
    indexWithinBlock(loci==j) = 1:blocksize(j);
end

fprintf('Start precomputation for Gaussian distr.\n');

for j = 1:L
    tmp = diag((1./diag(D{j})).^0.5)*(V{j}'*Zstat(loci==j,:));
    logGaussian0(j,:) = -sum(tmp.*tmp,1)/2;
    % logGaussian0(j,:) = -diag(Zstat(loci==j,:)'*V{j}*diag(1./diag(D{j}))*V{j}'*Zstat(loci==j,:))'/2;
end

% for i = 1:nsnp
%     mu = zeros(blocksize(loci(i)),nGWAS);
%     % mu(indexWithinBlock(i)) = ncp(loci(i)); % assign jth ncp to the ith mean
%     if opts.ncp == 1
%         tmp = Zstat(loci==loci(i),:);
%         % tmp = invLD{j}*tmp;
%         tmp(abs(tmp) < 3.7 ) = 3.7*sign(tmp(abs(tmp) < 3.7));
%     end
% 
%     if opts.ncp == 0 
%         tmp = ncp * ones(1,nGWAS);
%     end
%     mu(indexWithinBlock(i),:) = tmp;
%     mu = LD{j}*mu;
%     logGaussian(i,:) =  -diag((Zstat(loci==loci(i),:)-mu)'*invLD{j}*(Zstat(loci==loci(i),:)-mu))'/2 - logGaussian0(loci(i),:);
%     %muTrans = cholLD{loci(i)}*mu;
%     %logGaussian(i,:) = -sum((ZstatTrans(loci==loci(i),:)-muTrans*ones(1,nGWAS)).*  ...
%     %                    (ZstatTrans(loci==loci(i),:)-muTrans*ones(1,nGWAS)),1)/2 - logGaussian0(j,:);
% end
cumsumblock = cumsum(blocksize)-blocksize;
for j = 1:L
    Zstatj = Zstat(loci==j,:);
    if opts.ncp == 1
        tmp = Zstatj;
        %tmp = V{j}*diag(1./diag(D{j}))*V{j}'*Zstatj;
        tmp(abs(tmp) < ncp ) = ncp*sign(tmp(abs(tmp) < ncp));
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
%155453
sumGaussianDistr = zeros(L,nGWAS);
for j = 1:L
    sumGaussianDistr(j,:) = sum(exp(logGaussian(loci==j,:)),1)/blocksize(j);
end
%% all Pr(Z_jk|I_ijk = 1) are calculated compared to the baseline Pr(Z_jk|I_0jk = 1);

fprintf('Start EM steps.\n');

for iter = 2:opts.maxIters
     %% E-Step
     % update C1 posterior prob
     for j = 1:L
         %          C1(j,:) = ( Cpi1.*( sum(exp(logGaussian(loci==j,:)),1) )/blocksize(j) )./  ...
         %              ( Cpi1.*( sum(exp(logGaussian(loci==j,:)),1) )/blocksize(j) + (1-Cpi1) );
         C1(j,:) = ( Cpi1.*sumGaussianDistr(j,:) )./(Cpi1.*sumGaussianDistr(j,:) + (1-Cpi1) );
     end
     
    %M-Step: update Cpi1  
    Cpi1 = sum(C1,1)/L;
  
    
    Cpi1(Cpi1>0.999) = 0.999;
    Cpi1(Cpi1<0.001) = 0.001;

    % track penalized in-complete loglik
    
    llmat =  (ones(L,1)*Cpi1).* sumGaussianDistr + (1-(ones(L,1)*Cpi1));
    loglik(iter) = sum(sum(log(llmat)));
    
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

obj.loglik = loglik;
obj.I = I;
obj.C1 = C1;
obj.Cpi1 = Cpi1;