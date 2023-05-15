import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


def df_to_FT_labels(df, outname, class_colname, text_colname):
    with open(outname, 'w', encoding='utf-8') as fout:
        for index, row in df.iterrows():

            if isinstance(class_colname, list):
                s = ""
                for c in class_colname:
                    if row[c] == 1:
                        s += f'__label__{c} '

                s += str(row[text_colname])

            else:
                s = f'__label__{row[class_colname]} {row[text_colname]}'


            fout.write(s.replace('\n', ' ') + '\n')


def df_to_RNNLM_labels(df, outname, class_colname, text_colname, is_test, enrich=False, reverse=False):
    with open(outname, 'w', encoding='utf-8') as fout:
        for index, row in df.iterrows():
            s = row[text_colname]

            if reverse:
                s_lst = s.split()
                s_lst.reverse()
                s = ' '.join(s_lst)

            ground_truth = "" if is_test else f' {row[class_colname]}'
            s = f'{s} <eos>{ground_truth}'
            s = s.replace('\n', ' ')

            if enrich:
                s_lst = s.split()
                for i in range(5, len(s_lst), 6):
                    if len(s_lst) == 35 or i > (len(s_lst) - 2 ): # dont enrich too close to <eos>
                        break
                    s_lst.insert(i, ground_truth)
                s = ' '.join(s_lst)

            fout.write(s + '\n')


def split_df_dataset(df, ratio_val, ratio_test, text_colname, class_colname, save_path):

    X = df[text_colname]
    y = df[class_colname]

    if isinstance(class_colname, list):
        y = y.to_numpy()

    # Defines ratios, w.r.t. whole dataset.
    #ratio_val = 0.1
    #ratio_test = 0.2


    # Produces test split.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=ratio_test, random_state=42)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining, test_size=ratio_val_adjusted, random_state=42)

    df_train = pd.DataFrame()
    df_train[text_colname] = X_train
    df_train[class_colname] = y_train

    df_test = pd.DataFrame()
    df_test[text_colname] = X_test
    df_test[class_colname] = y_test

    df_valid = pd.DataFrame()
    df_valid[text_colname] = X_val
    df_valid[class_colname] = y_val

    df_to_FT_labels(df_train, f"{save_path}/FT_train.txt", class_colname, text_colname)
    df_to_FT_labels(df_test, f"{save_path}/FT_test.txt", class_colname, text_colname)
    df_to_FT_labels(df_valid, f"{save_path}/FT_valid.txt", class_colname, text_colname)

    #df_train_valid = pd.concat([df_train, df_valid])
    df_to_RNNLM_labels(df_train, f"{save_path}/RNN_train.txt", class_colname, text_colname, False)
    df_to_RNNLM_labels(df_test, f"{save_path}/RNN_test.txt", class_colname, text_colname, True)
    df_to_RNNLM_labels(df_valid, f"{save_path}/RNN_valid.txt", class_colname, text_colname, True)

    pd.to_pickle(df_train, f"{save_path}/train.p")
    pd.to_pickle(df_test, f"{save_path}/test.p")
    pd.to_pickle(df_valid, f"{save_path}/valid.p")











