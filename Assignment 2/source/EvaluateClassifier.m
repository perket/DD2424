function P = EvaluateClassifier(X, W, b)
    hiddenLayer = MakeHiddenLayer(X, W{1}, b{1});
    s2 = W{2}*hiddenLayer+b{2};
    P = exp(s2) ./ sum(exp(s2));
