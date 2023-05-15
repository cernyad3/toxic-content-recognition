import fasttext

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    model = fasttext.train_supervised(
        input="data/twitter/augmented/balanced/full/FT_train.txt",
        #autotuneValidationFile="data/twitter/preprocessed/multi/full/FT_valid.txt",
        #autotuneValidationFile="data/twitter/augmented/full/full//FT_valid.txt",
        #autotuneDuration=600,
        # autotuneModelSize="20M",
        epoch=1,
        #pretrainedVectors="model/unsupervised/wiki.simple.reduced_10.vec",
        bucket=1629546,
        minn=3,
        maxn=6,
        lr=0.05,
        wordNgrams=1,
        dim=10,

    )
    # save the model
    model.save_model("model/twitter/new_29_04/model.ftz")
    print("Testing model precision and recall at one")
    print(model.test("data/twitter/preprocessed/multi/full/FT_test.txt"))


    args_obj = model.f.getArgs()
    for hparam in dir(args_obj):
        if not hparam.startswith('__'):
            print(f"{hparam} -> {getattr(args_obj, hparam)}")


    # FT built in classification
    df = pd.read_pickle(f'data/twitter/preprocessed/multi/full/test.p')

    res = [i for i in list(df.columns) if i.startswith('tweet')][0]
    df["prediction"] = df[res].apply(lambda x: model.predict(str(x).replace("\n", " "))[0][0][-1:])
    y_pred = df['prediction'].to_numpy().astype(int)
    y_test = df['class'].to_numpy().astype(int)

    print(classification_report(y_test, y_pred, digits=3))
    print(confusion_matrix(y_test, y_pred, normalize='true'))




