function [X, Y, y] = LoadData(filename, mean_X)
    A = load(filename);
    X = double(A.data') / 255;
    y = A.labels';
    y = y + uint8(ones(1,length(y))); % Add one to simplify indexing
    
    X = X - repmat(mean_X, [1, size(X, 2)]);
    
    % Create image label matrix
    Y = zeros(10, length(X));
    for i = 1:length(Y)
        Y(y(i),i) = 1;
    end;