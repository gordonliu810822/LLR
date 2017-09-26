function [TrueSnp,Zstat,loci,y,SIGMA,RR] = generateData3(maf,L,M,rho,h2,n,nref, mag,OddsRatio,nGWAS)

SIGMA = zeros(M,M);
for j = 1:M
    for k= 1:M
        SIGMA(j,k) = rho^(abs(j-k));
    end
end

r = 2;
c1=[1;-1];
c2=[1;1];
v = [c1*ones(1,nGWAS/2), c2*ones(1,nGWAS/2)];
x = randn(L,r)*v;
x0 = ones(L,nGWAS)*log(OddsRatio);
prob = 1./(1+exp(-x*mag-x0));

y = binornd(1,prob);

loci = ones(M,1)*(1:L);
loci=loci(:);

nsnp = L*M;
TrueSnp = zeros(nsnp, nGWAS);

for k = 1:nGWAS
    for l = 1: L
        if y(l,k) == 1
            locusIndx = zeros(M,1);
            index = (M*(l-1)+1): (M*l);
            indx = randsample(M,1);
            locusIndx(indx) = 1;
            TrueSnp(index,k) = locusIndx;
        end
    end
end

b = zeros(nsnp,nGWAS);
for k = 1:nGWAS
    b(TrueSnp(:,k)==1,k) = randn(sum(TrueSnp(:,k)==1),1);
end

h2_true = zeros(nGWAS,1);
betah = zeros(nsnp,nGWAS);
s2 = zeros(nsnp,nGWAS);
% pvals = zeros(nsnp,nGWAS);

X00 = cell(L,1); Y  = cell(nGWAS,1); % Xo = cell(nGWAS,1);
% RRaw = cell(L,2); 
stderror = zeros(nGWAS,1);

for k = 1:nGWAS
    Y0 = 0;
    %Xo{k} = [];
    for l = 1:L
        index = (M*(l-1)+1): (M*l);
        AAprob = maf(index).^2.;
        Aaprob = 2*maf(index).*(1-maf(index));
        quanti = [1-Aaprob-AAprob; 1- AAprob]';
        X00{l} = mvnrnd(zeros(M,1),SIGMA,n);
        Xrefc = zeros(n,M);
        for j = 1:M
            cutoff = norminv(quanti(j,:));
            Xrefc(X00{l}(:,j) < cutoff(1),j) = 0;
            Xrefc(X00{l}(:,j) >= cutoff(1) & X00{l}(:,j) < cutoff(2),j) = 1;
            Xrefc(X00{l}(:,j) >= cutoff(2),j) = 2;
        end
        X00{l} = Xrefc;
        X = X00{l};%(X00{l} - repmat(mean(X00{l}),n,1))./repmat(std(X00{l}),n,1)/sqrt((alpha)*nsnp);
        Y0 = Y0 + X*b(loci==l,k);
        % RRaw{l,k} = corr(X00{l});
        % Xo{k} = [Xo{k},Xrefc];
    end
    stderror(k) = std(Y0)*sqrt((1-h2)/h2);
    e = stderror(k)*randn(n,1);%/mag(k);
    Y{k} = Y0 + e;
    
    h2_true(k) = var(Y0)/var(Y{k});
    for l = 1:L
        tmpbeta = zeros(M,1); tmps2 = zeros(M,1); tmppvals = zeros(M,1);
        for i = 1:M
            [betahat,se2,~,pval,~] = LinearRegCan2(X00{l}(:,i),Y{k});
            tmpbeta(i) = betahat(1);
            tmps2(i) = se2(1);
            tmppvals(i) = pval(1);
        end
        betah(loci==l,k) = tmpbeta;
        s2(loci==l,k) = tmps2;
        % pvals(loci==l,k) = tmppvals;
    end    
end

clear X00 Y stderror Xrefc X;

R = []; RR = cell(L,1);
for l = 1:L
    index = (M*(l-1)+1): (M*l);
    AAprob = maf(index).^2.;
    Aaprob = 2*maf(index).*(1-maf(index));
    quanti = [1-Aaprob-AAprob; 1- AAprob]';
    Xref = mvnrnd(zeros(M,1),SIGMA,nref);
    Xrefc = zeros(n,M);
    for j = 1:M
        cutoff = norminv(quanti(j,:));
        Xrefc(Xref(:,j) < cutoff(1),j) = 0;
        Xrefc(Xref(:,j) >= cutoff(1) & Xref(:,j) < cutoff(2),j) = 1;
        Xrefc(Xref(:,j) >= cutoff(2),j) = 2;
    end
    RR{l} = corr(Xrefc);
    % R = blkdiag(R,RR{l}); 
end

Zstat = betah./sqrt(s2);
clear Xref Xrefc;