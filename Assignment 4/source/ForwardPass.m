function [loss, a, h, p] = ForwardPass(RNN, X, Y, h0, n)
K = length(X);
m = length(h0);

p = zeros(K, n);    % p_1, p_2, .., p_n
h = zeros(m, n+1);  % h_0, h_1, .., h_n
a = zeros(m, n);    % a_1, a_2, .., a_n
h(:, 1) = h0;
loss = 0;

for t = 1:n
   a(:, t) = RNN.U * X(:, t) + RNN.W * h(:, t) + RNN.b;
   h(:, t+1) = tanh(a(:, t));
   ot = RNN.V * h(:, t+1) + RNN.c;
   p(:,t) = softmax(ot);
   loss = loss - log(Y(:,t)' * p(:,t));
end