function [model,details] = ssnnest(data,modelParam,options)
% Estimates state-space neural network model of a nonlinear dynamical
% system - currently SISO systems only
%
% INPUT:
% data: structure containing all the training data
% data.u:   training input: N x M x nu tensor, N = number of samples in one 
%           sequence, M = number of sequences, nu: number of inputs
% data.y:   training output: N x M x ny tensor, N = number of samples in  
%           one sequence, M = number of sequences, ny: number of outputs
%
% modelParam: strucure containing the parameters defining the model struct.
% modelParam.nu: number of inputs    
% modelParam.ny: number of outputs   
% modelParam.nx: number of states   
% modelParam.nn: number of neurons in the nonlinear state and output map
% modelParam.transferFcn: activation function for the nonlinear maps 
%                         ('tansig', 'radbas', 'poslin', ...)
%
% options: structure describing the estimation algorithm options
% options.nIter: number of iterations for the neural net training
% options.viewnet: display neural net (1) or not (0)
% options.feedthrough: estimate a direct feedthrough in the linear model
%                      (1) or not (0)
% options.transient: what type of transient do we consider: 'zero', 
%                    'periodic', 'none'
% options.nTran: how long of a transient do we consider, nTran < N
% options.initMethod: how should we initialize the model 'linear', 
%                     'random', 'model'
% options.modelInit: initial model in case the options 'model' is used.

%% extract options

% number of iterations in the network training
try nIter = options.nIter; catch; nIter = 1e3; end

% display the ss-nn network (1) or not (0)?
try options.viewnet; catch; options.viewnet = 0; end

% estimate feedthrough in linear model (1) or not (0)? The feedthrough is present
% by default in the current nn-ss programming
try options.feedthrough; catch; options.feedthrough = 1; end 

% OLD now embedded in the data structure
% % what type of transient do we consider: 'zero', 'periodic', 'none'
% try options.transient; catch; options.transient = 'none'; end 
% 
% % how long of a transient do we consider?
% try nTran = options.nTran; catch; nTran = 0; end

% initialization method
try options.initMethod; catch; options.initMethod  = 'linear'; end

% nonlinear output mapping (1) or purely linear output mapping (0)
try nloutput = options.nloutput; catch; nloutput = 1; end

%% extract model parameters

nu = modelParam.nu; % number of inputs
ny = modelParam.ny; % number of outputs
nx = modelParam.nx; % number of states
nnx = modelParam.nn; % number of neurons in the nonlinear state and output map

if nloutput
    nny = modelParam.nn; % number of neurons in the nonlinear state and output map
else
    nny=0;
end

transferFcn = modelParam.transferFcn; % 'tansig', 'radbas', 'poslin' - activation function for the nonlinear maps


%% prepare training data - SISO only
M = max(size(data));

[~,nu] = size(data{1}.u);
[~,ny] = size(data{1}.y);

% prepare data for nn training - MIMO extension needed
uTemp = [];
yTemp = [];
for ii=1:M
    N = length(data{ii}.u);
    switch data{ii}.transient
        case 'none'
            uTemp = [uTemp; data{ii}.u(:,1)];
            yTemp = [yTemp; NaN*ones(data{ii}.nTran,1); data{ii}.y(data{ii}.nTran+1:N,1)];
        case 'zero'
            uTemp = [uTemp; zeros(data{ii}.nTran,1); data{ii}.u(:,1)];
            yTemp = [yTemp; NaN*ones(data{ii}.nTran,1); data{ii}.y(:,1)];    
        case 'periodic'
            uTemp = [uTemp; data{ii}.u(N-data{ii}.nTran+1:N,1); data{ii}.u(:,1)];
            yTemp = [yTemp; NaN*ones(data{ii}.nTran,1); data{ii}.y(:,1)];
        case 'custom'
            uTemp = [uTemp; data{ii}.u(:,1)];
            yTemp = [yTemp; nan(size(data{ii}.y(:,1)))];
            yTemp(data{ii}.customMask) = data{ii}.y(data{ii}.customMask,1);
        otherwise
            error('ss-nn data prepartion error')
    end
end
uNN = num2cell(uTemp.');
yNN = num2cell(yTemp.');

% get the std of all combined inputs on which training takes place
iu = isnan(uTemp); % find NaN values
iy = isnan(yTemp);
iNan = iu | iy; % combine the NaN index vectors of u and y
uTemp(iNan) = []; % remove NaN values
uScale = std(uTemp); % compute std

%% estimate or load linear model (if linear init)



if strcmp(options.initMethod,'linear-custom') || strcmp(options.initMethod,'linear')
    % prepare data for initial linear model estimate - MIMO ready
    dataLin = iddata(squeeze(data{1}.y(:,:)),squeeze(data{1}.u(:,:)),1);
    if M>1
        for ii=2:M
            dataTemp = iddata(squeeze(data{ii}.y(:,:)),squeeze(data{ii}.u(:,:)),1);
            dataLin = merge(dataLin,dataTemp);
        end
    end
    if strcmp(options.initMethod,'linear-custom')
        % learn linear model on fake data - we just need the model
        % structure, the actual parameters are replaced later
        dataTemp = iddata(randn(1024,ny),randn(1024,nu),1); % generate fake data
        linMod = ssest(dataTemp, nx,'Ts',1,'Feedthrough',options.feedthrough,'Focus','simulation');

   
        linModCustom = options.linMod;
        
        linMod.A = linModCustom.A;
        linMod.B = linModCustom.B;
        linMod.C = linModCustom.C;
        linMod.D = linModCustom.D;
        
        options.initMethod = 'linear';
    else
        % learn linear model on the full dataset
        linMod = ssest(dataLin, nx,'Ts',1,'Feedthrough',options.feedthrough,'Focus','simulation');
    end
    
    [~,~,xCell] = sim(linMod,dataLin);

    if M>1
        % bring x from cell array to vector
        x = [];
        for ii=1:M
            x = [x; xCell{ii}];
        end
    else
        x = xCell;
    end

    % normalize the linear model such that the states have a unit variance
    T = diag(std(x))^-1;

    linMod.A = T*linMod.A*T^-1;
    linMod.B = T*linMod.B;
    linMod.C = linMod.C*T^-1;

    [yLinCell,~,~] = sim(linMod,dataLin);

    % uses old data structure
%     yLin = zeros(size(data.y));
%     if M>1
%         % bring x from cell array to vector
%         for ii=1:M
%             yLin(:,ii) = yLinCell.OutputData{ii};
%         end
%     else
%         yLin = yLinCell.OutputData;
%     end
% 
%     perfLTI = rms(data.y(:)-yLin(:))^2;
% 
%     disp(['Linear Model Performance: ' num2str(perfLTI)])
    
    % add to output
    details.linModel = linMod;
end

%% create state-space nn structure with explicit linear part
% two separate neural nets are used for the nonlinear state map and the
% nonlinear output map
%
% bias terms are added in every layer

numInputs = nu;
numLayers = 4;

biasConnect = ones(numLayers,1);
inputConnect = [ones(1,nu); ones(1,nu); ones(1,nu); ones(1,nu);];

layerConnect = zeros(numLayers,numLayers);
layerConnect(2,1) = 1; 
layerConnect(2,2) = 1;
layerConnect(3,2) = 1;
layerConnect(4,2) = 1;
layerConnect(4,3) = 1;
layerConnect(1,2) = 1;

outputConnect = [zeros(1,numLayers-1) 1];

net = network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);

% set the activation functions of the layers
net.layers{1}.transferFcn = transferFcn;
net.layers{2}.transferFcn = 'purelin'; 
net.layers{3}.transferFcn = transferFcn; 
net.layers{4}.transferFcn = 'purelin'; 

% set delay in the feedback loop
net.layerWeights{1,2}.delays = 1;
net.layerWeights{2,2}.delays = 1;
net.layerWeights{3,2}.delays = 1;
net.layerWeights{4,2}.delays = 1;
% set number of neurons per layer
net.layers{1}.size = nnx;
net.layers{2}.size = nx;
net.layers{3}.size = nny;
net.layers{4}.size = ny;

% set training setting - levenberg marquardt
net.trainFcn = 'trainlm';
net.adaptFcn = 'adaptwb';
net.plotFcns = {'plotresponse','plotperform'};
net.trainParam.epochs = nIter;
net.trainParam.min_grad = 1e-15;
net.trainParam.max_fail = 10;
net.trainParam.mu_dec = 0.5;
net.trainParam.mu_inc = 2;
net.trainParam.mu_max = 1e15;

netSSNN = net;

if options.viewnet
    view(netSSNN);
end

%% train network

% set up network
netTrainLin = init(netSSNN);
netTrainLin.inputs{1}.processFcns = {};
netTrainLin.outputs{2}.processFcns = {};
netTrainLin.trainParam.epochs = 0;
netTrainLin = train(netTrainLin,uNN,yNN);

switch lower(options.initMethod)
    case 'linear'
        % introduce the linear estimate - random inner weights & biases
        netTrainLin.IW{1} = rands(nnx,nu)/uScale; % works for 1-input only
        netTrainLin.IW{2} = linMod.B;
        netTrainLin.IW{3} = rands(nny,nu)/uScale; % works for 1-input only
        netTrainLin.IW{4} = linMod.D;

        netTrainLin.LW{1,2} = rands(nnx,nx);
        netTrainLin.LW{2,1} = zeros(nx,nnx);
        netTrainLin.LW{2,2} = linMod.A;
        netTrainLin.LW{3,2} = rands(nny,nx);
        netTrainLin.LW{4,2} = linMod.C;
        netTrainLin.LW{4,3} = zeros(ny,nny);

        netTrainLin.b{1} = rands(nnx,1);
        netTrainLin.b{2} = zeros(nx,1);
        netTrainLin.b{3} = rands(nny,1);
        netTrainLin.b{4} = zeros(ny,1);
    case 'random'
        netTrainLin.IW{1} = rands(nnx,nu)/uScale; % works for 1-input only
        netTrainLin.IW{2} = rands(nx,nu)/uScale;
        netTrainLin.IW{3} = rands(nny,nu)/uScale; % works for 1-input only
        netTrainLin.IW{4} = rands(ny,nu)/uScale;

        netTrainLin.LW{1,2} = rands(nnx,nx);
        netTrainLin.LW{2,1} = zeros(nx,nnx);
        netTrainLin.LW{2,2} = zeros(nx,nx); % too many unstable models otherwise
        netTrainLin.LW{3,2} = rands(nny,nx);
        netTrainLin.LW{4,2} = rands(ny,nx);
        netTrainLin.LW{4,3} = zeros(ny,nny);

        netTrainLin.b{1} = rands(nnx,1);
        netTrainLin.b{2} = zeros(nx,1);
        netTrainLin.b{3} = rands(nny,1);
        netTrainLin.b{4} = zeros(ny,1);
    case 'model'
        netTrainLin = options.modelInit;
    otherwise
        error('no initialization method specified')
end

ySimLinNN = sim(netTrainLin,uNN);
perfLin = perform(netTrainLin,yNN,ySimLinNN);
disp(['SS-NN Performance Init: ' num2str(perfLin)])

tic;
netTrainLin.trainParam.epochs = nIter;
[netTrainLin,trLin] = train(netTrainLin,uNN,yNN);
trainTimeLin = toc;

% simulate data
ySimLinNN = sim(netTrainLin,uNN);
perfLin = perform(netTrainLin,yNN,ySimLinNN);

disp(['SS-NN Performance Opt: ' num2str(perfLin) ', time elapsed: ' num2str(trainTimeLin)])

%% output

model = netTrainLin;
details.trainingDetails = trLin;

