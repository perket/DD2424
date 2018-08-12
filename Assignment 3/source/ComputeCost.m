function J = ComputeCost(X, Y, W, b, lambda, mu_exp, v_exp, epsilon, bn, mode)
    P = EvaluateClassifier(X, W, b, mu_exp, v_exp, epsilon, bn, mode);
    c = 0;
    [~,x] = size(X);
    for i = 1:x
        y = Y(:,i);
        p = P(:,i);
        c = c + -log(y'*p);
    end;
    J = c/x + lambda * sum(sum(W{1}.^2)) + lambda * sum(sum(W{2}.^2));
