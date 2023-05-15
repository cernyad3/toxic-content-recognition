import argparse
import time
import math
import os
import torch
import sys
import data
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from models import LSTM
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

parser = argparse.ArgumentParser(description='PyTorch implementation of RNN')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--data_reverse', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RHN, LSTM)')
parser.add_argument('--evaluate', default='', type=str, metavar='PATH',
                       help='path to pre-trained model (default: none)')
parser.add_argument('--evaluate_reverse', default='', type=str, metavar='PATH',
                       help='path to pre-trained model (default: none)')
parser.add_argument('--emsize', type=int, default=1000,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1000,
                    help='number of hidden units per layer')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--finetuning', type=int, default=100,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--lr', type=float, default=15,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--couple', action='store_true',
                    help='couple the transform and carry weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

def model_save(fn):
    torch.save(model.state_dict(), fn)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data, args.bptt)
# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz # length of data (1d) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.eval_batch_size)
test_data = batchify(corpus.test, args.eval_batch_size)

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)

#if args.model == 'LSTM':
model = LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)


total_params = sum(x.data.nelement() for x in model.parameters())
print('Model total parameters: {}'.format(total_params))

criterion = nn.CrossEntropyLoss()
###############################################################################
# Train and evaluate code
###############################################################################
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt): # iterate over batches
            data, targets = get_batch(data_source, i)

            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def evaluate_new(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt): # iterate over batches
            data, targets = get_batch(data_source, i)

            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

            label_idx = ((data == corpus.dictionary.word2idx["<eos>"]).nonzero(as_tuple=True)[0])[
                            0].item() + 1 - 1  # najde index výskytu <eos> (label je o 1 po)

            label = corpus.dictionary.idx2word[targets[label_idx]]
            label_indicies = ((targets == corpus.dictionary.word2idx[label]).nonzero(as_tuple=True)[0])

            total_loss += len(data) * criterion(output.view(-1, ntokens)[label_indicies], targets[label_indicies])
            hidden = model.init_hidden(1) # reinit hidden

    return total_loss / (len(data_source) - 1)


def evaluate_classif():
    print('=> testing...')
    y_pred = np.argmax(generate_labels(f"{args.data}/RNN_valid.txt"), axis=1)

    valid_df = pd.read_pickle(f"{args.data}/valid.p")

    valid_df["len"] = valid_df["tweet"].apply(lambda x: len(str(x).split()) + 1)  # +2 for <eos> and label
    valid_df = valid_df.loc[valid_df['len'] <= args.bptt]

    y_test = valid_df["class"].to_numpy().astype(int)

    target_names = ["hate", "offensive", "neither"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    macro_f1 = classification_report(y_test, y_pred, target_names=target_names, digits=3, output_dict=True)['macro avg']['f1-score']

    return float(macro_f1)


def generate_labels(path):
    labels = []
    model.eval()

    if args.cuda:
        model.cuda()

    with open(path, "r") as f:
        # READ SENTENCE
        for line in f:
            words = line.split()
            if len(words) > args.bptt:
                continue

            hidden = model.init_hidden(1)  # reinit hidden
            ids = []
            idss = []

            for word in words:
                ids.append(corpus.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))

            # flatten the array
            ids = torch.cat(idss)

            ids = batchify(ids, 1)
            input = ids.cuda()

            # GENERATE LABEL
            output, hidden = model(input, hidden)  # compute new hidden and output
            word_weights = output.squeeze().data.div(1).exp().cpu()  # squeeze output array

            # tady najit word_idx co odpovida 1 a 0 a srovnat pst
            best_label = ""
            best_weight = -1
            weights = []
            for i in range(3):
                w = word_weights[word_weights.size(0)-1][corpus.dictionary.word2idx[str(i)]]
                weights += [w]
                if w  > best_weight:
                    best_label = str(i)
                    best_weight = w

            # print(line, f' LABEL: {best_label}')
            # print(f"top weights: {torch.topk(word_weights[word_weights.size(0)-1], 3)}")
            # print(weights)
            labels.append(np.array(weights)) # tuple([int(best_label), weights])

    return np.array(labels)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)



        #print(list(map(lambda x: corpus.dictionary.idx2word[x], targets)))
        label_idx = ((data == corpus.dictionary.word2idx["<eos>"]).nonzero(as_tuple=True)[0])[0].item() + 1 - 1 # najde index výskytu <eos> (label je o 1 po)
        label = corpus.dictionary.idx2word[targets[label_idx]]
        label_indicies = ((targets == corpus.dictionary.word2idx[label]).nonzero(as_tuple=True)[0])
        loss = criterion(output.view(-1, ntokens)[label_indicies], targets[label_indicies])

        # print(label_idx)
        # print(corpus.dictionary.idx2word[targets[label_idx]])
        #
        # print(label_indicies)
        # print(list(map(lambda x: corpus.dictionary.idx2word[x], targets[label_indicies])))
        #
        # print(list(map(lambda x: corpus.dictionary.idx2word[x], targets)))
        #exit()

        hidden = model.init_hidden(args.batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            sys.stdout.flush()
###############################################################################
# Training
###############################################################################
if args.evaluate and args.evaluate_reverse:

    model.load_state_dict(torch.load(args.evaluate))
    probs_normal = generate_labels(f"{args.data}RNN_test.txt")

    model = LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    corpus = data.Corpus(args.data_reverse, args.bptt)
    model.load_state_dict(torch.load(args.evaluate_reverse))
    probs_reverse = generate_labels(f"{args.data_reverse}RNN_test.txt")

    y_pred = np.argmax((probs_normal + probs_reverse) / 2, axis=1)

    test_df = pd.read_pickle(f"../data/twitter/preprocessed/multi/rnn/test.p")
    res = [i for i in list(test_df.columns) if i.startswith('tweet')][0]
    test_df["len"] = test_df[res].apply(lambda x: len(str(x).split()) + 1)  # +1 for label
    test_df = test_df.loc[test_df['len'] <= args.bptt]

    y_test = test_df["class"].to_numpy().astype(int)

    target_names = ["hate", "offensive", "neither"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, index=["hate", "offensive", "neither"],
                         columns=["hate", "offensive", "neither"])
    # plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    figure = sns.heatmap(df_cm, annot=True, cmap="Blues", cbar=False).get_figure()
    figure.savefig('rnn_multi_model_heatmap.eps', format='eps')
    figure.savefig('rnn_multi_model_heatmap.png', format='png')

elif args.evaluate: # evaluate single model
    print("=> loading checkpoint '{}'".format(args.evaluate))
    model.load_state_dict(torch.load(args.evaluate))
    print('=> testing...')
    y_pred = np.argmax(generate_labels(f"{args.data}RNN_test.txt"), axis=1)

    test_df = pd.read_pickle(f"../data/twitter/preprocessed/multi/rnn/test.p")

    res = [i for i in list(test_df.columns) if i.startswith('tweet')][0]
    test_df["len"] = test_df[res].apply(lambda x: len(str(x).split()) + 1)  # +1 for label
    test_df = test_df.loc[test_df['len'] <= args.bptt]

    y_test = test_df["class"].to_numpy().astype(int)

    target_names = ["hate", "offensive", "neither"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, index=["hate", "offensive", "neither"],
                         columns=["hate", "offensive", "neither"])
    # plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    figure = sns.heatmap(df_cm, annot=True, cmap="Blues", cbar=False).get_figure()
    figure.savefig('rnn_single_model_heatmap.eps', format='eps')
    figure.savefig('rnn_single_model_heatmap.png', format='png')

else:
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000
    training_start_time = time.time()
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)

        # Loop over epochs.
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                val_loss2 = evaluate_new(val_data)
                evaluate_classif()
                #val_loss2 = evaluate_classif()

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                print(f"{type(val_loss2)} . {type(stored_loss)}")
                print(f"{val_loss2} < {stored_loss} ?")
                if True:#val_loss2 < stored_loss:
                    model_save(f"model_{epoch}.pt")
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate_new(val_data)
                evaluate_classif()
                #val_loss = evaluate_classif()
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                print(f"{type(val_loss)} . {type(stored_loss)}")
                print(f"{val_loss} < {stored_loss} ?")
                if True: # val_loss < stored_loss:
                    model_save(f"model_{epoch}.pt")
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'adam':
                    scheduler.step(val_loss)

                #if loss did increase for n steps or this is last epoch
                if (args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono]))) or (args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and epoch == args.epochs-1):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                best_val_loss.append(val_loss)

            print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model.load_state_dict(torch.load(args.save))

    # Run on test data.
    test_loss = evaluate_new(test_data)
    train_time = training_start_time - time.time()
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | time {:5.2f}s'.format(
        test_loss, math.exp(test_loss), train_time))
    print('=' * 89)
    sys.stdout.flush()
