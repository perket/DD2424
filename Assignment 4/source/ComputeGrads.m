function grads = ComputeGrads(X, Y, RNN, a, h, p, n)
    m = length(h);
    
    grad_o = (p - Y)';
    grads.c = (sum(grad_o))';
    grads.V = grad_o' * h(:,2 : end)';
    
    grad_h = grad_o(n, :) * RNN.V;
    
    grad_a = zeros(n, m);
    grad_a(n, :) = grad_h * diag(1 - (tanh(a(:, n))).^2);
    
    for t = n-1 : -1 : 1
       grad_h = grad_o(t, :) * RNN.V + grad_a(t+1, :) * RNN.W;
       grad_a(t, :) = grad_h * diag(1 - (tanh(a(:, t))).^2);
    end
    
    grads.b = (sum(grad_a))';
    
    grads.U = grad_a' * X';
    grads.W = grad_a' * h(:,1 : end-1)';
    
    