clear all;
clc;
close all;
format longEng

addpath Datasets/cifar-10-batches-mat;

% Parameters
n_batch = 50; %
n_epochs = 5; %
h = 1e-5; % 
nodes_in_hidden_layer = 50; % Number of nodes in hidden layer

% Hyper parameters
eta = .025; % learning rate
lambda = .00998; % regularization
decay_rate = .998; % decay in learning rate
rho = .8; % Momentum

% Data setup
[X,Y,y,mean_X] = LoadBatch('data_batch_1.mat');
[XValid, YValid, yValid] = LoadData('data_batch_2.mat', mean_X);
[XTest,YTest,yTest] = LoadData('test_batch.mat', mean_X);
[XBatches, YBatches] = GetMiniBatches(X, Y, n_batch);

[W, b] = InitModel(X,nodes_in_hidden_layer);

[W,b,costs_train,costs_test,accs_train,accs_test] = trainingLoop(XBatches,YBatches,W,b,n_epochs,eta,lambda,rho,decay_rate,nodes_in_hidden_layer,X,Y,y,XTest,YTest,yTest);

PTest = EvaluateClassifier(XTest, W, b);
acc_test = ComputeAccuracy(PTest, yTest);

fprintf('Accuracy test\t%f\tCost test\t', acc_test);
for n = 1:length(costs_test)
    fprintf('%f\t ', costs_test(n));
end;

P = EvaluateClassifier(X, W, b);
acc_train = ComputeAccuracy(P, y);

fprintf('Accuracy train\t%f\tCost train\t', acc_train);
for n = 1:length(costs_train)
    fprintf('%f\t ', costs_train(n));
end;


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