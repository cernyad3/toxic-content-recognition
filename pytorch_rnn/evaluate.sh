#!/bin/bash

python3 main.py --data ../data/twitter/augmented/balanced/rnn/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 0.3 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 10 --save twitter_rnn.pt --log-interval 1000 --evaluate best_model.pt  --evaluate_reverse best_reverse_model.pt --data_reverse ../data/twitter/augmented/balanced/rnn_reverse/

# HYPERPAMETERS TO TRY

# model = [LSTM, RNN_RELU, RNN_TANH]
# lr = [20, 30] / [0.3, 1]
# padding zmenšít (cca 20)
# dropout = [0, 0.2, 0.4]
# hidden = [100, 200, 400]