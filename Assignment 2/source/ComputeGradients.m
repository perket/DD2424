function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda, nodes_in_hidden_layer)
    [hY, ~] = size(Y);
    [hX, lX] = size(X);
    
    grad_W = {zeros(nodes_in_hidden_layer, hX), zeros(hY, nodes_in_hidden_layer)};
    grad_b = {zeros(nodes_in_hidden_layer, 1), zeros(hY, 1)};
    
    hidden_layer = MakeHiddenLayer(X, W{1}, b{1});
    
    for i = 1:lX
        yy = Y(:,i);
        xx = X(:,i);
        
        p = P(:,i);
        [hp,~] = size(p);
        diag_p = eye(hp) .* p;
        g = -((yy' / (yy' * p)) * (diag_p - p * p'));
        
        grad_b{2} = grad_b{2} + g';
        grad_W{2} = grad_W{2} + g' * hidden_layer(:,i)';
        
        s1 = W{1} * xx + b{1};
        [hs1,~] = size(s1);
        
        g = g * W{2};
        diag_s1 = eye(hs1) .* (s1>0);
        g = g * diag_s1;
        
        grad_b{1} = grad_b{1} + g';
        grad_W{1} = grad_W{1} + g' * xx';
    end;
    
    grad_b{2} = grad_b{2} ./ lX;
    grad_W{2} = grad_W{2} ./ lX + 2 * lambda .* W{2};
    
    grad_b{1} = grad_b{1} ./ lX;
    grad_W{1} = grad_W{1} ./ lX + 2 * lambda .* W{1};