function [loglik_masked, ave_crit] = cv_Zscore_lowrank_boosting(Zstat, LD, ncpEst, loci, obj0, opts, nfold)
L = length(unique(loci));
nGWAS = size(Zstat,2);
crossSet = zeros(L,nGWAS);


% for iter = 1:100
%     if iter > 1
%         if min(sum(crossSet,2)) > nGWAS*0.6;
%             break;
%         end
%     end
for k = 1:nGWAS
    crossSet(:,k) = crossvalind('Kfold', L, nfold);
end
% end


loglik_masked = zeros(opts.maxIters,nfold);

for ifold = 1:nfold
    fprintf('Start %d-th fold:\n', ifold);
    mask = ones(L,nGWAS);
    mask(crossSet == ifold) = 0;
    tmp = Zscore_lowrank(Zstat, loci, LD, ncpEst,obj0, mask,opts);
    loglik_masked(:, ifold) = tmp.loglik_masked;
end
 
ave_crit = mean(loglik_masked,2);
