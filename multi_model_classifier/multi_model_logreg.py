import sent_vectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import fasttext

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("hello")


    train_df = pd.read_pickle("../data/twitter/augmented/balanced/full/train.p")
    #train_df = pd.read_pickle("../data/twitter/augmented/full/full/train.p")
    #train_df = pd.read_pickle("../data/twitter/preprocessed/multi/full/train.p")


    #valid_df = pd.read_pickle("../data/twitter/augmented/balanced/simple/valid.p")
    #train_df = pd.concat([train_df, valid_df], axis=0)

    #train_df = pd.read_pickle("../data/twitter/preprocessed/multi/train.p")
    test_df =  pd.read_pickle("../data/twitter/preprocessed/multi/full/test.p")

    # test_df = pd.read_pickle("../data/twitter/labeled_data.p")
    # test_df["tweet"] = test_df["tweet"].apply(preprocess_tweet)
    print(list(train_df.columns))
    test_df['tweet'] = test_df["tweet_full"].copy()
    train_sentences = train_df['tweet'].apply(lambda x: str(x).replace('\n', ' ')).to_numpy()
    test_sentences = test_df['tweet'].apply(lambda x: str(x).replace('\n', ' ')).to_numpy()

    X_train = np.array(spe.embed_sentences(train_sentences))
    y_train = train_df['class'].to_numpy().astype(int)

    X_test = np.array(spe.embed_sentences(test_sentences))
    y_test = test_df['class'].to_numpy().astype(int)

    print(X_train.shape)
    print(y_train.shape)


    print(X_test.shape)
    print(y_test.shape)



    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    target_names = ["hate", "offensive", "neither"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    print(type(classification_report(y_test, y_pred, target_names=target_names, digits=3, output_dict=True)['macro avg']['f1-score']))
    #print(confusion_matrix(y_test, y_pred, normalize='true'))

    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, index=["hate", "offensive", "neither"],
                         columns=["hate", "offensive", "neither"])
    plt.figure(figsize=(10, 7))
    figure = sns.heatmap(df_cm, annot=True, cmap="Blues", cbar=False).get_figure()
    figure.savefig('multi_model_heatmap.eps', format='eps')








