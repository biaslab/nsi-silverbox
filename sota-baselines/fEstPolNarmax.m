function model = fEstPolNarmax(data,options)

u = data.u;
y = data.y;
N = length(u);

nd = options.nd; % maximal polynomial degree
nb = options.nb; % maximal input delay
na = options.na; % maximal output delay
nk = nb+na+1;

%% generate polynomial combinations

comb = [0:nd];
for ii=2:nk
    comb = [repmat(comb,1,nd+1); kron([0:nd],ones(1,size(comb,2)))];
    
    % remove combinations which have degree higher than nd
    ndComb = sum(comb);
    comb = comb(:,ndComb<=nd);
end
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

%% perform prediction error NARX estimate
theta = K\y;

%% save model
model.nb = nb;
model.na = na;
model.comb = comb;
model.theta = theta;

