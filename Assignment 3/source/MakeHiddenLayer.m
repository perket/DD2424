function [hiddenLayer, s0, mu, v, mu_exp, v_exp] = MakeHiddenLayer(X, W, b, mu_exp, v_exp, epsilon, bn, mode)
    s = W * X + b;
    s0 = s;
    mu = 0;
    v = 0;
    if bn == 1
        [s, mu, v] = BatchNormalize(s, mu_exp, v_exp, epsilon, mode); 
        
        if strcmp(mode, 'train')
            alpha = .99;
            mu_exp = alpha * mu + (1 - alpha) * mu;
            v_exp = alpha * v + (1 - alpha) * v;
        end;
    end;
    hiddenLayer = max(0,s);
    
    