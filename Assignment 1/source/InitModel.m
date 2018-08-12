function [W, b] = InitModel(X)
    [d, ~] = size(X);
    W = normrnd(0,.01,10,d);
    b = normrnd(0,.01,10,1);