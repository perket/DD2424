function P = EvaluateClassifier(X, W, b)
    s = W*X+b;
    P = exp(s) ./ sum(exp(s));