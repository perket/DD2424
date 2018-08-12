
function [W,b,costs_train,costs_test,accs_train,accs_test] = trainingLoop(XBatches,YBatches,W,b,n_epochs,eta,lambda,rho,decay_rate,nodes_in_hidden_layer,X,Y,y,XTest,YTest,yTest)
    accs_train = double.empty();
    accs_test = double.empty();
    costs_train = double.empty();
    costs_test = double.empty();

    %Momentum
    moment_W = {zeros(size(W{1})), zeros(size(W{2}))};
    moment_b = {zeros(size(b{1})), zeros(size(b{2}))};

    [~,~,l] = size(XBatches);
    
    for i = 1:n_epochs
        for j = 1:l
            % Get j:th batch
            XBatch = XBatches(:,:,j)';
            YBatch = YBatches(:,:,j)';

            %hidden_layer = MakeHiddenLayer(XBatch, W{1}, b{1});
            P = EvaluateClassifier(XBatch, W, b);

            [grad_W, grad_b] = ComputeGradients(XBatch, YBatch, P, W, b, lambda, nodes_in_hidden_layer);
            
            % Calculate momentum
            moment_W{1} = eta * grad_W{1} + rho * moment_W{1};                                                                                        
            moment_b{1} = eta * grad_b{1} + rho * moment_b{1};                                                                                        
            moment_W{2} = eta * grad_W{2} + rho * moment_W{2};                                                                                        
            moment_b{2} = eta * grad_b{2} + rho * moment_b{2};

            % Update W's and b's
            W{1} = W{1} - moment_W{1};
            b{1} = b{1} - moment_b{1};
            W{2} = W{2} - moment_W{2};
            b{2} = b{2} - moment_b{2};

        end;
        eta = decay_rate * eta;

        %hidden_layer = MakeHiddenLayer(X, W{1}, b{1});
        P = EvaluateClassifier(X, W, b);
        acc_train = ComputeAccuracy(P, y);
        cost_train = ComputeCost(X, Y, W, b, lambda);

        %hidden_layer = MakeHiddenLayer(XTest, W{1}, b{1});
        PTest = EvaluateClassifier(XTest, W, b);
        acc_test = ComputeAccuracy(PTest, yTest);
        cost_test = ComputeCost(XTest, YTest, W, b, lambda);

        accs_train = [accs_train acc_train];
        accs_test = [accs_test acc_test ];

        costs_train = [costs_train cost_train];
        costs_test = [costs_test cost_test ];

        % Displays the cost and accuracy of each epoch
        fprintf('Cost train: %f\n', cost_train);
        fprintf('Accuracy train: %f\n', acc_train);
        fprintf('Cost test: %f\n', cost_test);
        fprintf('Accuracy test: %f\n\n', acc_test);

        fprintf('Epoch %f is done.\n', i);
    end;