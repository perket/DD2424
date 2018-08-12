function [grad_W, grad_b, mu_exp, v_exp] = ComputeGradients(X, Y, P, W, b, lambda, nodes_in_hidden_layers, mu_exp, v_exp, epsilon, bn, mode)
    [hY, ~] = size(Y);
    [hX, lX] = size(X);
    
    n = length(nodes_in_hidden_layers);
    
    grad_W = cell(n+1,1);
    grad_b = cell(n+1,1);
    
    grad_W{1} = zeros(nodes_in_hidden_layers(1), hX);
    grad_b{1} = zeros(nodes_in_hidden_layers(1), 1);
    
    for i = 2:n
        grad_W{i} = zeros(nodes_in_hidden_layers(i), nodes_in_hidden_layers(i-1));
        grad_b{i} = zeros(nodes_in_hidden_layers(i), 1);
    end;
    
    grad_W{n+1} = zeros(hY, nodes_in_hidden_layers(n));
    grad_b{n+1} = zeros(hY, 1);
    
    layers = cell(n+1,1);
    layers{1} = X;
    
    if bn == 1
        mu = cell(n-1,1);
        v = cell(n-1,1);
        S = cell(n-1,1);
    end;
    
    for i = 2:n+1
        if bn == 1
            [layers{i}, S{i-1}, mu{i-1}, v{i-1}, mu_exp{i-1}, v_exp{i-1}] = MakeHiddenLayer(layers{i-1}, W{i-1}, b{i-1}, mu_exp{i-1}, v_exp{i-1}, epsilon, bn, mode);
        else
            [layers{i}, ~, ~, ~, ~, ~] = MakeHiddenLayer(layers{i-1}, W{i-1}, b{i-1}, v_exp{i-1}, epsilon, bn, mode);
        end;
    end;
    
    if bn == 1
        gs = zeros(lX, nodes_in_hidden_layers(n));
        % Last layer gradients
        for i = 1:lX
            yy = Y(:,i);
            p = P(:,i);

            [hp,~] = size(p);
            diag_p = eye(hp) .* p;
            g = -((yy' / (yy' * p)) * (diag_p - p * p'));
            
            grad_W{n+1} = grad_W{n+1} + g' * layers{n+1}(:,i)';
            grad_b{n+1} = grad_b{n+1} + g';
            
            g = g * W{n+1};
            s = S{n}(:,i);
            [hs, ~] = size(s);
            diag_s = eye(hs) .* (s>0);
            g = g * diag_s;
            
            gs(i,:) = g;
        end;
        
        grad_W{n+1} = grad_W{n+1} ./ lX + 2 * lambda .* W{n+1};
        grad_b{n+1} = grad_b{n+1} ./ lX;
        
        for i = n:-1:1
            x = layers{i};
            gs = BatchNormBackPass(gs, S{i}, mu{i}, v{i}, epsilon);
            
            grad_W{i} = gs' * x' ./ lX + 2 * lambda .* W{i};
            grad_b{i} = sum(gs,1)' ./ lX;
            
            gs = gs * W{i};
            if i > 1
                for j = 1:lX
                    s = S{i-1}(:,j);
                    [hs, ~] = size(s);
                    diag_s = eye(hs) .* (s>0);
                    gs(j,:) = gs(j,:) * diag_s;
                end;
            end;
        end;
    else
        for i = 1:lX
            yy = Y(:,i);
            p = P(:,i);

            [hp,~] = size(p);
            diag_p = eye(hp) .* p;
            g = -((yy' / (yy' * p)) * (diag_p - p * p'));

            for k = n+1:-1:2


                grad_b{k} = grad_b{k} + g';
                grad_W{k} = grad_W{k} + g' * layers{k}(:,i)';

                g = g * W{k};

                s = W{k-1} * layers{k-1}(:,i) + b{k-1};
                [hs,~] = size(s);
                diag_s = eye(hs) .* (s>0);

                g = g * diag_s;
            end;

            grad_b{1} = grad_b{1} + g';
            grad_W{1} = grad_W{1} + g' * layers{1}(:,i)';
        end;
        for i = 1:n+1
            grad_b{i} = grad_b{i} ./ lX;
            grad_W{i} = grad_W{i} ./ lX + 2 * lambda .* W{i};
        end;
    end;
    
    