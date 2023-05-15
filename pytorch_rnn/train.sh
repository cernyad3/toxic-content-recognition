#!/bin/bash

python3 main.py --data ../data/twitter/augmented/balanced/rnn_reverse/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 0.2 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 20 --save twitter_rnn.pt --log-interval 1000

# HYPERPAMETERS TO TRY

# model = [LSTM, RNN_RELU, RNN_TANH]
# lr = [20, 30] / [0.3, 1]
# dropout = [0, 0.2, 0.4]
# hidden = [100, 200, 400]


# RESULTS

#python3 main.py --data ../data/twitter/augmented/balanced/rnn/ --optimizer sgd --model RNN_TANH --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 1 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 10 --save twitter_rnn.pt --log-interval 1000
# epoch 9 - f1 0.675, acc 0.818

# python3 main.py --data ../data/twitter/augmented/balanced/rnn/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 1 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 10 --save twitter_rnn.pt --log-interval 1000
# epoch 10 - f1 0.679, acc 0.835
    # switching nhid to 400 did not help at all, same result at epoch 10

# python3 main.py --data ../data/twitter/augmented/balanced/rnn/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 0.3 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 10 --save twitter_rnn.pt --log-interval 1000
# epoch 4 - f1 0.688, acc 0.843

# python3 main.py --data ../data/twitter/augmented/balanced/rnn/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 0.6 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0.3 --epochs 10 --save twitter_rnn.pt --log-interval 1000
# epoch 10 - f1 0.685, acc 0.845

# python3 main.py --data ../data/twitter/augmented/balanced/rnn_reverse/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 0.2 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 10 --save twitter_rnn.pt --log-interval 1000
# and
# python3 main.py --data ../data/twitter/augmented/balanced/rnn/ --optimizer sgd --model RNN_RELU --cuda --nonmono 5 --batch_size 1 --bptt 35 --lr 0.3 --clip 0.25 --seed 1111 --emsize 200  --nhid 200  --nlayers 2 --dropout 0 --epochs 10 --save twitter_rnn.pt --log-interval 1000
# normal epoch 4, reverse epoch 10 - f1 0.696, acc 0.848

