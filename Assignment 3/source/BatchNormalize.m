function [s, mu, v] = BatchNormalize(s, mu, v, epsilon, mode)
    if strcmp(mode, 'train')
        mu = mean(s, 2);
        v = var(s, 0, 2);
    end;
    
    s2 = s - mu;
    v2 = eye(length(v)) .* (v + epsilon);
    v2 = (v2 ^ (-1/2));
    s = v2 * s2;