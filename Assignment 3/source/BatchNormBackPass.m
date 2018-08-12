function g = BatchNormBackPass(g, s, mu ,v , epsilon)
    [n, m] = size(g);
    for i = 1:n
        diags = eye(m) .* (s(:,i) - mu)';
        diagv = eye(m) .* (v + epsilon);
        gradv = -sum(g(i,:) * (diagv^(-3/2)) * diags, 1) / 2;
        gradmu = -sum(g(i,:) * (diagv^(-1/2)), 1);
        g(i,:) = g(i,:) * (diagv^(-1/2)) + 2 * (gradv * diags + gradmu) / n;
    end;