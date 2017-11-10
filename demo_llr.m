clear;
% rng(1000);
L = 500; M = 20; %screen 1
mag = 3; nGWAS = 20; OddsRatio = 1/49;
n = 5000; LDfolder = 'tmp1';
rho = 0.5; h2 = 0.5;

wd = 'G:\Data\GoogleDrive\Work\GPA_matlab\OtherWork\LowRank\Z_score\Duke-NUS\Allfiles\packages_llr';%wd = '/home/svu/gmsjl/LLR/LLR2';
nstep = 3000; eps = 0.1;nref = 400;
% wdPAINTOR = '~/Work/LLR';
cd(wd);
maf = load('maf_10000.txt');
nsnp = L*M;
rng(1000);

[TrueSnp,Zstat,loci,y,~,RR] = generateData3(maf,L,M,rho,h2,n,nref, mag,OddsRatio,nGWAS);
Cpio = sum(y,1)/L;
    
LD = cell(L,1);
for j = 1:L
    LD{j} = RR{j};
end

%%LLR
options = GPA_lowrank_set(nGWAS,[]);
options.eps = eps;
options.maxIters = nstep;
options.ncp = 1;

obj0 = Zscore_init(Zstat, loci, LD, 3.7,options);
Cpi1 = obj0.Cpi1;

nfold =5;
[loglik_masked, ave_crit] = cv_Zscore_lowrank_boosting(Zstat, LD, 3.7, loci, obj0, options, nfold);
save('ave_crit.txt','ave_crit','-ascii','-tabs');
save('loglik_masked.txt','loglik_masked','-ascii','-tabs');

[~,ind] = max(ave_crit);
index = 1:nstep;
optstep = min(index(ave_crit > ave_crit(ind) - 5));

options.maxIters = optstep; mask = ones(L,nGWAS);
obj = Zscore_lowrank(Zstat, loci, LD, 3.7, obj0, mask,options);
x = obj.x;
[~,~,V] = svd(x, 'econ');

subplot(2,2,1)
plot(obj.Dall')
plot(ave_crit)
scatter(V(1,:),V(:,2));
Traits = num2cell(1:20);
text(V(1,:)+0.02, V(:,2)+0.02, Traits);

d=pdist(V(:,1:2));
z=linkage(d);
dendrogram(z);
