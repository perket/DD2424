function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    [hY, lY] = size(Y);
    [hX, lX] = size(X);
    grad_W = zeros(hY,hX);
    grad_b = zeros(hY,1);
    
    for i = 1:lX
        yy = Y(:,i);
        x = X(:,i);
        p = P(:,i);
        
        diag_p = eye(10) .* p;
        g = -((yy' / (yy' * p)) * (diag_p - p * p'));
        
        grad_b = grad_b + g';
        grad_W = grad_W + g' * x';
    end;
    
    grad_b = grad_b ./ lX;
    grad_W = grad_W ./ lX + 2 * lambda .* W;