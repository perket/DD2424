
function [W,b,costs_train,costs_test,accs_train,accs_test] = TrainingLoop(XBatches,YBatches,W,b,n_epochs,eta,lambda,rho,epsilon,decay_rate,nodes_in_hidden_layers,X,Y,y,XTest,YTest,yTest,bn,mode)
    accs_train = double.empty();
    accs_test = double.empty();
    costs_train = double.empty();
    costs_test = double.empty();
    
    n = length(nodes_in_hidden_layers);

    %Momentum
    moment_W = cell(n+1,1);
    moment_b = cell(n+1,1);
    mu_exp = cell(n+1,1);
    v_exp = cell(n+1,1);
    
    for i = 1:n+1
        moment_W{i} = zeros(size(W{i}));
        moment_b{i} = zeros(size(b{i}));
    end;

    [~,~,l] = size(XBatches);
    
    for i = 1:n_epochs
        for j = 1:l
            % Get j:th batch
            XBatch = XBatches(:,:,j)';
            YBatch = YBatches(:,:,j)';

            %hidden_layer = MakeHiddenLayer(XBatch, W{1}, b{1});
            [P, mu_exp, v_exp] = EvaluateClassifier(XBatch, W, b, mu_exp, v_exp, epsilon, bn, mode);

            [grad_W, grad_b, mu_exp, v_exp] = ComputeGradients(XBatch, YBatch, P, W, b, lambda, nodes_in_hidden_layers, mu_exp, v_exp, epsilon, bn, mode);
            
            % Gradient checking
%             [ngrad_b, ngrad_W] = ComputeGradsNum(XBatch, YBatch, W, b, lambda, 1e-5);
%             
%             diff_W = cell(n+1, 1);
%             diff_b = cell(n+1, 1);
%             
%             for m = 1:n+1
%                 diff_W{m} = abs(grad_W{m} - ngrad_W{m});
%                 diff_b{m} = abs(grad_b{m} - ngrad_b{m});
%                 fprintf('grad_W{%i} min: %e, max: %e, avg: %e\n', m,min(diff_W{m}(:)),max(diff_W{m}(:)),mean(diff_W{m}(:)));
%                 fprintf('grad_b{%i} min: %e, max: %e, avg: %e\n', m,min(diff_b{m}(:)),max(diff_b{m}(:)),mean(diff_b{m}(:)));
%             end;
            
            % Calculate momentum
            for q = 1:n+1
                moment_W{q} = eta * grad_W{q} + rho * moment_W{q};                                                                                        
                moment_b{q} = eta * grad_b{q} + rho * moment_b{q};
            end;

            % Update W's and b's
            for q = 1:n+1
                W{q} = W{q} - moment_W{q};
                b{q} = b{q} - moment_b{q};
            end;
            
        end;
        eta = decay_rate * eta;

        %hidden_layer = MakeHiddenLayer(X, W{1}, b{1});
        P = EvaluateClassifier(X, W, b, mu_exp, v_exp, epsilon, bn, mode);
        acc_train = ComputeAccuracy(P, y);
        cost_train = ComputeCost(X, Y, W, b, lambda, mu_exp, v_exp, epsilon, bn, mode);

        %hidden_layer = MakeHiddenLayer(XTest, W{1}, b{1});
        PTest = EvaluateClassifier(XTest, W, b, mu_exp, v_exp, epsilon, bn, mode);
        acc_test = ComputeAccuracy(PTest, yTest);
        cost_test = ComputeCost(XTest, YTest, W, b, lambda, mu_exp, v_exp, epsilon, bn, mode);

        accs_train = [accs_train acc_train];
        accs_test = [accs_test acc_test ];

        costs_train = [costs_train cost_train];
        costs_test = [costs_test cost_test ];

        % Displays the cost and accuracy of each epoch
        fprintf('Epoch %i is done.\n', i);
        
        fprintf('Cost train: %f\n', cost_train);
        fprintf('Accuracy train: %f\n', acc_train);
        fprintf('Cost test: %f\n', cost_test);
        fprintf('Accuracy test: %f\n\n', acc_test);
    end;