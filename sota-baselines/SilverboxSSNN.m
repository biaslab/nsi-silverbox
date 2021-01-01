clear all
close all
clc

%% set variables
fs = 610.35; % Hz
iTrain = 4.05*1e4:131072;
iTest = 1:(4.05*1e4-1);
timeTrain = [0:(length(iTrain)-1)]/fs;

%% load data
dataBenchmark = load('../data/SNLS80mV.mat');
y = dataBenchmark.V2(iTrain);
u = dataBenchmark.V1(iTrain);
yVal = dataBenchmark.V2(iTest);
uVal = dataBenchmark.V1(iTest);

%% optimze NL model
nTran = 1024;
nIter = 1e3;

data{1}.u = u';
data{1}.y = y';
data{1}.transient = 'zero';
data{1}.nTran = nTran; % the whole first part is neglected as transient

modelParam.nu = 1;
modelParam.ny = 1;
modelParam.nx = 2;
modelParam.nn = 5;
modelParam.transferFcn = 'tansig';

options.nIter = nIter;
options.viewnet = 0;
options.feedthrough = 1;

[model,details] = ssnnest(data,modelParam,options);

%% plot results
yVal = dataBenchmark.V2;
uVal = dataBenchmark.V1;

uNN = num2cell(uVal);
yModNN = sim(model,uNN);
yMod = cell2mat(yModNN).'; 

figure; hold on;
plot(yVal)
plot(yMod)
plot(yVal(:)-yMod(:))
legend('system','model','error')

%% Plot error

error = (yVal(:)-yMod(:)).^2;

figure;
semilogy(error)
legend('error')

figure;
scatter(1:length(error), error, '.', 'black')
set(gca, 'YScale', 'log')
ylabel('error')
xlabel('time (t)')
savefig('SilverboxSSNN-error2.fig')


