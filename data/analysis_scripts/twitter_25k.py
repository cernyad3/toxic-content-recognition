import pandas as pd

if __name__ == "__main__":
    # df = pd.read_pickle('../twitter/labeled_data.p')
    #
    # print(df["class"].value_counts(normalize=True))
    #
    # df = pd.read_csv('../twitter_100k/hatespeech_text_label_vote_RESTRICTED_100K.csv', sep='\t', names=["text", "class", "n"])
    #
    #
    # print(df["class"].value_counts(normalize=True))
    #
    # df = pd.read_csv('../wiki/train.csv', header=0)
    #
    # # "toxic","severe_toxic","obscene","threat","insult","identity_hate"
    #
    # print(df["toxic"].value_counts(normalize=True))
    # print(df["severe_toxic"].value_counts(normalize=True))
    # print(df["obscene"].value_counts(normalize=True))
    # print(df["threat"].value_counts(normalize=True))
    # print(df["insult"].value_counts(normalize=True))
    # print(df["identity_hate"].value_counts(normalize=True))
    #
    # df["non_toxic"] = df.iloc[:,2:8].apply(lambda x: 1 if (sum(x)==0) else 0, axis=1)
    #
    # print(df["non_toxic"].value_counts(normalize=True))
    # print(df.shape[0])

    df = pd.read_pickle('../twitter/augmented/balanced/rnn/train.p')

    df["len"] = df["tweet"].apply(lambda x : len(str(x).split()))

    print(df["len"].quantile([.01, .1, .25, .5, .75, 0.95, 0.99, 1]))