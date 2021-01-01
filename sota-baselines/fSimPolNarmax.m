function ySim = fSimPolNarmax(data,model)

u = data.u;
N = length(u);

nb = model.nb;
na = model.na;
nk = nb+na+1;

comb = model.comb;
nComb = size(comb,2);

%% simulation
nc = max(nb,na);
uSim = [zeros(nc,1); u(:)]; % zeropadding for unknown initial conditions
ySim = zeros(size(uSim));
for ii=nc+1:N+nc
    % construct regressor vector
    KLin = [uSim(ii:-1:ii-nb); ySim(ii-1:-1:ii-na)].'; % row vector

    K = ones(1,nComb);
    for kk=1:nComb
        for jj=1:nk
            K(1,kk) = K(1,kk).*KLin(1,jj).^comb(jj,kk);
        end
    end
    ySim(ii) = K*model.theta;
end
ySim = ySim(nc+1:end); %remove zero padding part    