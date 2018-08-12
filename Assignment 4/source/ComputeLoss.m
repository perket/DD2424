function loss = ComputeLoss(X, Y, RNN, h)
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
n = size(X, 2);
loss = 0;

for t = 1 : n
    at = W*h + U*X(:, t) + b;
    h = tanh(at);
    o = V*h + c;
    pt = exp(o);
    p = pt/sum(pt);

    loss = loss - log(Y(:, t)'*p);
end