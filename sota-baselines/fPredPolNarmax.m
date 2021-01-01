function yPred = fPredPolNarmax(data,model)

u = data.u;
y = data.y;
N = length(u);

nb = model.nb;
na = model.na;
nk = nb+na+1;

comb = model.comb;
nComb = size(comb,2);

%% generate regressor matrix

KLin = zeros(N,nk);
for ii=0:nb
    KLin(:,ii+1) = [zeros(ii,1); u(1:end-ii)];
end
for ii=1:na
    KLin(:,nb+1+ii) = [zeros(ii,1); y(1:end-ii)];
end

K = ones(N,nComb);
for ii=1:nComb
    for jj=1:nk
        K(:,ii) = K(:,ii).*KLin(:,jj).^comb(jj,ii);
    end
end

%% 1-step ahead prediction

yPred = K*model.theta;