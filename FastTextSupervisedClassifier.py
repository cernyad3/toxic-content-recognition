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


def df_to_RNNLM_labels(df, outname, class_colname, text_colname, is_test, enrich=False):
    with open(outname, 'w', encoding='utf-8') as fout:
        for index, row in df.iterrows():
            ground_truth = "" if is_test else f' {row[class_colname]}'
            s = f'{row[text_colname]} <eos>{ground_truth}'
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
    #df = pd.read_pickle('data/twitter/preprocessed/labeled_data.p')

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


if __name__ == '__main__':
    #df = pd.read_csv('/mnt/c/Users/AdamC/Plocha/Skola/bachelors_thesis/data/wiki/preprocessed/train.csv')
    #df_to_FT_labels(df, '/mnt/c/Users/AdamC/Plocha/Skola/bachelors_thesis/data/wiki/preprocessed/multi/FT_whole.txt',
    #                ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'non_toxic'],
    #                'comment_text')
    #split_df_dataset(df, 0.1, 0.2, 'comment_text', ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'non_toxic'],
    #                                              '/mnt/c/Users/AdamC/Plocha/Skola/bachelors_thesis/data/wiki/preprocessed/multi')

    #model = fasttext.train_supervised(input='data/wiki/preprocessed/multi/FT_train_large.txt',
    #                                  autotuneValidationFile='data/wiki/preprocessed/multi/FT_valid.txt', loss='ova')
    #model.save_model('model/wiki/multi_kaggle.bin')


    # model = fasttext.load_model(
    #     '/mnt/c/Users/AdamC/Plocha/Skola/bachelors_thesis/model/wiki/wiki_small.ftz')
    # df = pd.read_csv("submission1.csv")
    #
    # df1 = pd.read_csv("data/wiki/preprocessed/test_kaggle.csv")
    #
    # y_pred = pd.merge(df1, df, on="id")
    #
    # y_pred = y_pred.drop(columns=["id", "comment_text"])
    # y_pred = y_pred.to_numpy()
    #
    # y_pred[y_pred.astype(float) >= 0.5] = 1
    # y_pred[y_pred.astype(float) < 0.5] = 0
    # y_pred = y_pred.astype(int)
    #
    # print(y_pred)
    # print(y_pred.shape)
    #
    # y_true = pd.read_csv("data/wiki/ground_truth.csv")
    # y_true = y_true.drop(columns=["id"])
    # y_true = y_true.to_numpy()
    #
    # print(y_true)
    # print(y_true.shape)
    #
    # print(classification_report(y_true, y_pred))
    #
    # y_true = np.c_[y_true, np.zeros(y_true.shape[0])]
    # y_pred = np.c_[y_pred, np.zeros(y_pred.shape[0])]
    #
    # #print(y_true)
    # #print(y_true[np.sum(y_true, axis=1) == 0, -1])
    #
    # y_true[np.sum(y_true, axis=1) == 0, -1] = 1
    # y_pred[np.sum(y_pred, axis=1) == 0, -1] = 1
    #
    # print(y_pred)
    # print(y_true)
    #
    #
    # print(classification_report(y_true, y_pred))

    # model = fasttext.load_model(
    #     '/mnt/c/Users/AdamC/Plocha/Skola/bachelors_thesis/model/wiki/wiki_small.ftz')
    #
    # df = pd.read_csv('/mnt/c/Users/AdamC/Plocha/Skola/bachelors_thesis/data/wiki/preprocessed/test.csv')
    #
    # submission = pd.DataFrame(columns=["id", "toxic", "severe_toxic", "threat", "insult", "obscene", "identity_hate"])
    #
    # for index, row in df.iterrows():
    #     predition = model.predict(str(row["comment_text"]).replace('\n', ' '), k=-1)
    #     prob_dict = dict(zip(map(lambda x: x.split("__label__")[1], predition[0]), predition[1]))
    #     submission.loc[len(submission.index)] = [row["id"], f'{prob_dict["toxic"]:.6f}',
    #                                              f'{prob_dict["severe_toxic"]: .6f}', f'{prob_dict["obscene"]:.6f}',
    #                                              f'{prob_dict["threat"]:.6f}', f'{prob_dict["insult"]:.6f}',
    #                                              f'{prob_dict["identity_hate"]:.6f}']
    #
    # #submission.drop(columns=["comment_text"])
    # submission.to_csv("submission.csv")
    models = [
        'model/twitter/twitter_small.ftz',
        'model/twitter/small_new.ftz',
        'model/twitter/small_augmented_balanced.ftz',
        'model/twitter/small_augmented_full.ftz',
        'model/twitter/small_augmented_balanced_new.ftz'
        ]

    for path in models:
        print(path)
        model = fasttext.load_model(path)

        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")

        print('-'*50)









