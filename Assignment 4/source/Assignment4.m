clear all;
clc;
close all;
format longg

%% 0.1 Read in the data

book_fname = 'Datasets/Goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = lower(fscanf(fid,'%c'));
fclose(fid);

book_chars = unique(book_data);
char_cnt = length(book_data);
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i = 1:length(book_chars)
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end;

%% 0.2 Set hyper-parameters & initialize the RNN's parameters

m = 100;
K = length(book_chars);
eta = .1;
seq_length = 25;
sig = .01;

RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

ada.b = zeros(size(RNN.b));
ada.c = zeros(size(RNN.c));
ada.U = zeros(size(RNN.U));
ada.W = zeros(size(RNN.W));
ada.V = zeros(size(RNN.V));

%% training

epochs = 20;
losses = [];
smooth_loss = -log(1 / length(book_chars)) * seq_length;
lowest_loss = smooth_loss;
best_RNN = RNN;
iterations = round(char_cnt / seq_length - .5);

for e = 1 : epochs
    for i = 1 : iterations
        X = zeros(K,seq_length);
        Y = zeros(K,seq_length);
        
        X_chars = book_data((i-1)*seq_length+1:i*seq_length);
        Y_chars = book_data((i-1)*seq_length+2:i*seq_length+1);

        for p = 1:seq_length
            X(char_to_ind(X_chars(p)),p) = 1;
            Y(char_to_ind(Y_chars(p)),p) = 1;
        end;
        
        if isequal(mod(i, 10000), 1)
            h0 = zeros(m,1);
            synth = Synthesize(RNN, X(:,1), h0, 200);
        end
        
        [loss, a, h, p] = ForwardPass(RNN, X, Y, h0, seq_length);
        [RNN, ada] = BackwardPass(RNN, ada, X, Y, a, h, p, seq_length);
        
        smooth_loss = .999 * smooth_loss + .001 * loss;
        
        if smooth_loss < lowest_loss
            lowest_loss = smooth_loss;
            best_RNN = RNN;
        end
        
        if isequal(mod(i, 10000), 1)
            fprintf('epoch: %d,iteration: %d smooth_loss: %f\n',e,i,smooth_loss);
            c = [];
            for j = 1 : length(synth)
                c = [c ind_to_char(synth(j))];
            end
            disp(c);
        end
        
        if isequal(mod(i, 1000), 1)
            losses = [losses smooth_loss];
        end
    end
end

%% Display 1000 synthesized characters
fprintf('Best model, smooth_loss = %f\n',lowest_loss);
h0 = zeros(m,1);
synth = Synthesize(best_RNN, X(:,1), h0, 1000);
c = [];
for j = 1 : length(synth)
    c = [c ind_to_char(synth(j))];
end
disp(c);

%% Plot smooth_loss
% Plots evolution of the cost
figure(1);
x = 1:1:epochs * round(iterations / 1000 + .5);
plot(x,losses,'b');
title('Cost')
legend('Smooth loss')
