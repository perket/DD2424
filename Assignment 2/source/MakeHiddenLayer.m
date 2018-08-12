function hiddenLayer = MakeHiddenLayer(X, W, b)
    s = W * X + b;
    hiddenLayer = max(0,s);