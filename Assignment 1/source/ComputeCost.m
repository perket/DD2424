function J = ComputeCost(X, Y, W, P, lambda)
    c = 0;
    [~,x] = size(X);
    for i = 1:x
        y = Y(:,i);
        p = P(:,i);
        c = c + -log(y'*p);
    end;
    J = c/x + lambda * sum(sum(W.^2));
    %F = -log(Y'*P);
    %d = (-log(Y*P'));
    %isequal(c,d)
    %J = sum(sum(-log(Y'*P))) / x + lambda * sum(sum(W.^2));
