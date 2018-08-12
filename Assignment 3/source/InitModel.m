function [W, b] = InitModel(X, nodes_in_hidden_layers)
    [d, ~] = size(X);
    
    n = length(nodes_in_hidden_layers);
    W = cell(n+1,1);
    b = cell(n+1,1);
    
    b{1} = zeros(nodes_in_hidden_layers(1),1);
    W{1} = normrnd(0,.001,nodes_in_hidden_layers(1),d);
    
    for i = 2:n
        b{i} = zeros(nodes_in_hidden_layers(i),1);
        W{i} = normrnd(0,.001,nodes_in_hidden_layers(i),nodes_in_hidden_layers(i-1));
    end;
    
    W{n+1} = normrnd(0,.001,10,nodes_in_hidden_layers(n));
    b{n+1} = zeros(10,1);