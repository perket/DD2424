function gradCheck(batch_size, X, Y, RNN, dh, m, K)

% numerical gradients
n_grads = ComputeGradsNum(X(:, 1 : batch_size), Y(:, 1 : batch_size), RNN, dh);

% analytical gradients
h0 = zeros(size(RNN.W, 1), 1);
[loss, a, h, p] = ForwardPass(RNN, X(:, 1 : batch_size), Y(:, 1 : batch_size), h0, seq_length);
[~, a, h, ~, p] = forward_Pass(RNN, X(:, 1 : batch_size), Y(:, 1 : batch_size), h0, batch_size, K, m);
grads = ComputeGrads(RNN, X(:, 1 : batch_size), Y(:, 1 : batch_size), a, h, p, batch_size, m);

% relative error rate
eps = 1e-5;

for f = fieldnames(RNN)'
    num_g = n_grads.(f{1});
    ana_g = grads.(f{1});
    denominator = abs(num_g) + abs(ana_g);
    numerator = abs(num_g - ana_g);
    gradcheck_max = max(numerator(:))/max(eps, sum(denominator(:)));
    gradcheck_sum = sum(numerator(:))/max(eps, sum(denominator(:)));
    disp(['Field name: ' f{1}]);
    disp(['max error: ' num2str(gradcheck_max) ', sum error: ' num2str(gradcheck_sum)]);
end

end