function [RNN, ada] = BackwardPass(RNN, ada, X, Y, a, h, p ,n)
    grads = ComputeGrads(X, Y, RNN, a, h, p, n);

%     Gradient check
%     dh = 1e-4;
%     num_grads = ComputeGradsNum(X, Y, RNN, dh);
%     for f = fieldnames(RNN)'
%         num_g = num_grads.(f{1});
%         ana_g = grads.(f{1});
%         
%         diff = abs(num_g - ana_g);
%         max_diff = max(diff(:));
%         avg_diff = mean(diff(:));
%         rel_diff = abs(num_g - ana_g)./abs(num_g + ana_g);
%         rel_diff(isnan(rel_diff)) = 0;
%         max_rel_diff = max(rel_diff(:));
%         avg_rel_diff = mean(rel_diff(:));
%         
%         fprintf('Field name: %s, max diff: %d, average diff: %d, max relative diff: %d, average relative diff: %d\n', f{1}, max_diff, avg_diff, max_rel_diff, avg_rel_diff);
%     end
    
    eta = .1;
    eps = 1e-8;
    
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    
    for f = fieldnames(RNN)'
        ada.(f{1}) = ada.(f{1}) + grads.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta * (grads.(f{1}) ./ sqrt(ada.(f{1}) + eps));
    end;