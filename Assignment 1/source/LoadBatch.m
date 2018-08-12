function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = double(A.data') / 255;
    y = A.labels';
    y = y + uint8(ones(1,length(y))); % Add one to simplify indexing
    
    % Create image label matrix
    Y = zeros(10, length(X));
    for i = 1:length(Y)
        Y(y(i),i) = 1;
    end;