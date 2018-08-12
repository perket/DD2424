clear all;
clc;
close all;

addpath Datasets/cifar-10-batches-mat/;

% Creates empty arrays to store results.
costs_test = double.empty();
costs_train = double.empty();
accs_train = double.empty();
accs_test = double.empty();

% Parameters
lambda = 0.01;
n_batch = 500;
n_epochs = 10;

% Learning rate with linear step size
eta = .01;
final_eta = .001;
eta_step = (eta - final_eta)/n_epochs;

% Data setup
[X,Y,y] = LoadBatch('data_batch_1.mat');
[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat');
[W, b] = InitModel(X);
[XBatches, YBatches] = GetMiniBatches(X, Y, n_batch);
[~,~,l] = size(XBatches);

% Learning loop
for i = 1:n_epochs
    for j = 1:l
        % Get j:th batch
        XBatch = XBatches(:,:,j)';
        YBatch = YBatches(:,:,j)';
        
        % Make prediction and calculate gradients
        P = EvaluateClassifier(XBatch, W, b);
        [grad_W, grad_b] = ComputeGradients(XBatch, YBatch, P, W, lambda);
        
        % Update W and b
        W = W - eta * grad_W;
        b = b - eta * grad_b;
        eta = eta - eta_step;
    end;
    
    
    % Calculate predictions, cost and accuracy of each epoch
    P_train = EvaluateClassifier(X, W, b);
    P_test = EvaluateClassifier(Xtest, W, b);
    cost_test = ComputeCost(Xtest, Ytest, W, P_test, lambda);
    cost_train = ComputeCost(X, Y, W, P_train, lambda);
    acc_train = ComputeAccuracy(P_train, y);
    acc_test = ComputeAccuracy(P_test, ytest);
    
    % Displays the cost and accuracy of each epoch
    fprintf('Cost train: %f\n', cost_train);
    fprintf('Accuracy train: %f\n', acc_train);
    fprintf('Cost test: %f\n', cost_test);
    fprintf('Accuracy test: %f\n\n', acc_test);
    
    % Store cost and accuracy of each epoch
    costs_test = [costs_test cost_test];
    costs_train = [costs_train cost_train];
    accs_train = [accs_train acc_train];
    accs_test = [accs_test acc_test ];
end;

% Plots the weights
figure(1);
for i=1:10
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [2,5]);

% Plots evolution of the cost
figure(2);
x = 1:1:n_epochs;
plot(x,costs_test,'g',x,costs_train,'b');
title('Cost')
legend('Validation','Training')

% Plots evolution of the accuracy
figure(3);
x = 1:1:n_epochs;
plot(x,accs_test,'g',x,accs_train,'b');
title('Accuracy')
legend('Validation','Training')
