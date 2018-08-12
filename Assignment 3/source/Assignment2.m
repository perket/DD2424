clear all;
clc;
close all;
format longEng

addpath Datasets/cifar-10-batches-mat;

validation_size = 1000;

% Parameters
n_batch = 50; %
n_epochs = 5; %
h = 1e-5; % 
nodes_in_hidden_layer = 100; % Number of nodes in hidden layer

% Hyper parameters
eta = .044576;%.044576; % learning rate
lambda = .00998;%.009980; % regularization
decay_rate = .998; % decay in learning rate
rho = .8;%.8; % Momentum

% Data setup
[X,Y,y,mean_X] = LoadBatch('data_batch_1.mat');
[XValid, YValid, yValid] = LoadData('data_batch_2.mat', mean_X);
[XTest,YTest,yTest] = LoadData('test_batch.mat', mean_X);
[XBatches, YBatches] = GetMiniBatches(X, Y, n_batch);

% validation_size = 1000;
% xl = length(X);
% XTest = X(:,xl-validation_size+1:xl);
% YTest = Y(:,xl-validation_size+1:xl);
% yTest = y(:,xl-validation_size+1:xl);
% X = X(:,1:xl-validation_size);
% Y = Y(:,1:xl-validation_size);
% y = y(:,1:xl-validation_size);


% Hyper paramets search
% Coarse search
% etas = [.01 ,.025, .05 ,.1, .25, .5, .75];
% lambdas = [.001, .0025, .005, .01, .025, .05, .1];
% Best
% eta	0.025	lambda	0
% eta	0.025	lambda	0.001
% eta	0.025	lambda	0.0025

% Fine search
% etas = (.04-.01).*rand(1,4) + .01; % (.04-.01).*rand(1,8) + .01;
% lambdas = (.005-.001).*rand(1,6) + .001; % lambdas = (.005-0).*rand(1,8) + 0;
% Best
% eta	0.022175	lambda	0.001514
% eta	0.022175	lambda	0.003633
% eta	0.016228	lambda	0.002828
% eta	0.016228	lambda	0.001514
% eta	0.016228	lambda	0.002218

params = [[.025; 0.0025]];%[[.019748; .015289], [.012757; .015289], [.022719; .015289], [.035; .002]];%,[.038103; .018084], [.010577; .020221], [.041772; .018084]];

for param = params
    eta = param(1);
    lambda = param(2);
%for eta = etas
%    for lambda = lambdas
        [W, b] = InitModel(X,nodes_in_hidden_layer);
        
        [W,b,costs_train,costs_test,accs_train,accs_test] = trainingLoop(XBatches,YBatches,W,b,n_epochs,eta,lambda,rho,decay_rate,nodes_in_hidden_layer,X,Y,y,XTest,YTest,yTest);
        fprintf('\neta\t%f\tlambda\t%f\t', eta, lambda);
        %hidden_layer = MakeHiddenLayer(XTest, W{1}, b{1});
        PTest = EvaluateClassifier(XTest, W, b);
        acc_test = ComputeAccuracy(PTest, yTest);

        fprintf('Accuracy test\t%f\tCost test\t', acc_test);
        for n = 1:length(costs_test)
            fprintf('%f\t ', costs_test(n));
        end;
        %hidden_layer = MakeHiddenLayer(X, W{1}, b{1});
        P = EvaluateClassifier(X, W, b);
        acc_train = ComputeAccuracy(P, y);

        fprintf('Accuracy train\t%f\tCost train\t', acc_train);
        for n = 1:length(costs_train)
            fprintf('%f\t ', costs_train(n));
        end;
%    end;
end;



% %% Plots the weights1
% figure(1);
% for i=1:10
%     im = reshape(W1(i, :), 32, 32, 3);
%     s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
%     s_im{i} = permute(s_im{i}, [2, 1, 3]);
% end
% montage(s_im, 'Size', [2,5]);
% 
% %% Plots the weights2
% figure(1);
% for i=1:10
%     im = reshape(W2(i, :), 32, 32, 3);
%     s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
%     s_im{i} = permute(s_im{i}, [2, 1, 3]);
% end
% montage(s_im, 'Size', [2,5]);

% Plots evolution of the cost
figure(2);
x = 1:1:n_epochs;
plot(x,costs_test,'g',x,costs_train,'b');
title('Cost')
legend('Test','Training')

% Plots evolution of the accuracy
figure(3);
x = 1:1:n_epochs;
plot(x,accs_test,'g',x,accs_train,'b');
title('Accuracy')
legend('Test','Training')