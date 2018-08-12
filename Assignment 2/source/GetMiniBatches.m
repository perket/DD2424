function [XBatches, YBatches] = GetMiniBatches(Xtrain, Ytrain, n_batch)
    [d,N] = size(Xtrain);
    [K,~] = size(Ytrain);
    
    XBatches = zeros(N/n_batch,d,n_batch);
    YBatches = zeros(N/n_batch,K,n_batch);
    
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = Xtrain(:, j_start:j_end);
        Ybatch = Ytrain(:, j_start:j_end);
        
        XBatches(j,:,:) = Xbatch;
        YBatches(j,:,:) = Ybatch;
    end
    
    % Permute to simplify picking out image representations from the
    % matrices.
    XBatches = permute(XBatches,[3 2 1]);
    YBatches = permute(YBatches,[3 2 1]);
