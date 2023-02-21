import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split


def df_to_FT_labels(df, outname, class_colname, text_colname):
    with open(outname, 'w', encoding='utf-8') as fout:
        for index, row in df.iterrows():
            fout.write(f'__label__{row[class_colname]} {row[text_colname]}'.replace('\n', ' ') + '\n')


def split_df_dataset():
    df = pd.read_pickle('data/twitter/preprocessed/labeled_data.p')

    X = df['tweet']
    y = df['class_binary']

    # Defines ratios, w.r.t. whole dataset.
    ratio_val = 0.1
    ratio_test = 0.2

    # Produces test split.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=ratio_test)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining, test_size=ratio_val_adjusted)

    df_train = pd.DataFrame()
    df_train['tweet'] = X_train
    df_train['class_binary'] = y_train

    df_test = pd.DataFrame()
    df_test['tweet'] = X_test
    df_test['class_binary'] = y_test

    df_valid = pd.DataFrame()
    df_valid['tweet'] = X_val
    df_valid['class_binary'] = y_val

    df_to_FT_labels(df_train, "data/twitter/preprocessed/FT_train_binary.txt", "class_binary", "tweet")
    df_to_FT_labels(df_test, "data/twitter/preprocessed/FT_test_binary.txt", "class_binary", "tweet")
    df_to_FT_labels(df_valid, "data/twitter/preprocessed/FT_valid_binary.txt", "class_binary", "tweet")

    pd.to_pickle(df_train, "data/twitter/preprocessed/train_binary.p")
    pd.to_pickle(df_test, "data/twitter/preprocessed/test_binary.p")
    pd.to_pickle(df_valid, "data/twitter/preprocessed/valid_binary.p")


if __name__ == '__main__':
    model = fasttext.train_supervised(input='data/twitter/preprocessed/FT_train_binary.txt', autotuneValidationFile='data/twitter/preprocessed/FT_valid_binary.txt')
    print(model.test("data/twitter/preprocessed/FT_test_binary.txt"))
    model.save_model('model_twitter_binary.bin')

    #model = fasttext.load_model('model_twitter.bin')
    #df = pd.read_pickle('data/twitter/preprocessed/labeled_data.p')
    #df["prediction"] = df["tweet"].apply(lambda x: model.predict(x)[0][0][-1])

    #print(df.head())




