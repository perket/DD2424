function synth_indexes = Synthesize(RNN, xt, ht, n)
%% 0.3 Synthesize text from your randomly initialized RNN
    synth_indexes = int16.empty(n,0);
    K = length(xt);
    for t = 1:n
        at = RNN.W * ht + RNN.U * xt + RNN.b;
        ht = tanh(at);
        ot = RNN.V * ht + RNN.c;
        pt = softmax(ot);

        cp = cumsum(pt);
        a = rand;
        ixs = find(cp-a > 0);
        ii = ixs(1);
        
        xt = zeros(K,1);
        xt(ii) = 1;
        
        synth_indexes(t) = ii;
    end;