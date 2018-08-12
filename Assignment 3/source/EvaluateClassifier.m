function [P, mu_exp, v_exp] = EvaluateClassifier(X, W, b, mu_exp, v_exp, epsilon, bn, mode)
    hiddenLayer = X;
    k = length(W);
    
    for i = 1:k-1
        [hiddenLayer, ~, ~, ~, mu_exp{i}, v_exp{i}] = MakeHiddenLayer(hiddenLayer, W{i}, b{i}, mu_exp{i}, v_exp{i}, epsilon, bn, mode);
    end;
    s = W{k}*hiddenLayer+b{k};
    P = exp(s) ./ sum(exp(s));
