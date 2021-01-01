% Silverbox Polynomial NARX Model Estimation

close all
clear all
clc

%% set variables
fs = 610.35; % Hz
iTrain = 4.05*1e4:131072;
iTest = 1e3:(4.05*1e4-1); % start at 1e3 to avoid transient
iTestNoExtra = 1e3:(3*1e4); % start at 1e3 to avoid transient
timeTrain = [0:(length(iTrain)-1)]/fs;

nb = 2; % # input delays
na = 2; % # output delays
nd = 3; % # degree polynomial nonlinearity

%% load data
dataBenchmark = load('../data/SNLS80mV.mat');

yTot = dataBenchmark.V2;
uTot = dataBenchmark.V1;

yTrain = dataBenchmark.V2(iTrain);
uTrain = dataBenchmark.V1(iTrain);

yTest = dataBenchmark.V2(iTest);
uTest = dataBenchmark.V1(iTest);

%% estimate
data.u = uTrain(:);
data.y = yTrain(:);
options.nb = nb;
options.na = na;
options.nd = nd;
% model = fEstPolNarmax(data,options);
model = fEstMonNarmax(data,options);

%% validate
data.u = uTot(:);
data.y = yTot(:);
yPred = fPredPolNarmax(data,model);
ySim = fSimPolNarmax(data,model);


%% plot

figure; hold on;
plot(yTot)
plot((yTot(:)-ySim(:)))
plot((yTot(:)-yPred(:)))
legend('system output','simulation error','prediction error')

disp('1-Step Ahead Prediction')
disp(['RMS Training Error: ' num2str(rms(yTot(iTrain).'-yPred(iTrain)))])
disp(['RMS Test Error: ' num2str(rms(yTot(iTest).'-yPred(iTest)))])
disp(['RMS Test Error - no extrapolation: ' num2str(rms(yTot(iTestNoExtra).'-yPred(iTestNoExtra)))])
RMS_pred = rms(yTot(iTest).'-yPred(iTest));

disp('Simulation')
disp(['RMS Training Error: ' num2str(rms(yTot(iTrain).'-ySim(iTrain)))])
disp(['RMS Test Error: ' num2str(rms(yTot(iTest).'-ySim(iTest)))])
disp(['RMS Test Error - no extrapolation: ' num2str(rms(yTot(iTestNoExtra).'-ySim(iTestNoExtra)))])
RMS_sim = rms(yTot(iTest).'-ySim(iTest));


