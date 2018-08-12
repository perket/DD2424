function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

P = EvaluateClassifier(X, W, b);
c = ComputeCost(X, Y, W, P, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    P = EvaluateClassifier(X, W, b_try);
    c2 = ComputeCost(X, Y, W, P, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    P = EvaluateClassifier(X, W_try, b);
    c2 = ComputeCost(X, Y, W_try, P, lambda);
    
    grad_W(i) = (c2-c) / h;
end