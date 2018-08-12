function [W, b] = InitModel(X, nodes_in_hidden_layer)
    [d, ~] = size(X);
    W = {normrnd(0,.001,nodes_in_hidden_layer,d), normrnd(0,.001,10,nodes_in_hidden_layer)};
    b = {zeros(nodes_in_hidden_layer,1), zeros(10,1)};
    %W1 = normrnd(0,.001,nodes_in_hidden_layer,d);
    %W2 = normrnd(0,.001,10,nodes_in_hidden_layer);
    
    %W1 = .1 * ones(nodes_in_hidden_layer,d);
    %W2 = .1 * ones(10,nodes_in_hidden_layer);
    
    %b1 = zeros(nodes_in_hidden_layer,1);
    %b2 = zeros(10,1);