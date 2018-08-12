function acc = ComputeAccuracy(P, y)
    [~,prediction] = max(P);
    acc = sum(prediction==y) / length(P);