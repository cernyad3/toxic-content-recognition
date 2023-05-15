import fasttext
import itertools

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    preprocessing_options = ["none", "simple", "full"]
    data_options = ["original", "balanced", "full"]
    dim_options = [10, 100]
    pretrained_vectors_options = [True, False]

    path_dict = {
        "original" : "data/twitter/preprocessed/multi",
        "balanced" : "data/twitter/augmented/balanced",
        "full" : "data/twitter/augmented/full"
    }

    options = [preprocessing_options, data_options, dim_options, pretrained_vectors_options]
    options = list(itertools.product(*options))
    print(options)
    print(f"total combinations: {len(options)}")


    for option in options:
        print("-"*50)
        print(option)

        preprocessing = option[0]
        data = option[1]
        dim = option[2]
        pretrained = option[3]

        pretrained_vec = f"model/unsupervised/wiki.simple.reduced_{dim}.vec"

        model = fasttext.train_supervised(
            input=f"{path_dict[data]}/{preprocessing}/FT_train.txt",
            autotuneValidationFile=f"{path_dict[data]}/{preprocessing}/FT_valid.txt",
            autotuneDuration=600,
            autotuneModelSize="50M",
            dim=dim,
            pretrainedVectors=(pretrained_vec if pretrained else ""),
        )

        # save the model
        model.save_model(f"model/twitter/gridsearch3/{data}_{preprocessing}_{dim}_{pretrained}.ftz")
        print("Testing model precision and recall at one")
        print(model.test(f"data/twitter/preprocessed/multi/{preprocessing}/FT_test.txt"))

        # print model params
        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")

        df = pd.read_pickle(f'data/twitter/preprocessed/multi/{preprocessing}/test.p')

        res = [i for i in list(df.columns) if i.startswith('tweet')][0]
        df["prediction"] = df[res].apply(lambda x: model.predict(str(x).replace("\n", " "))[0][0][-1:])
        y_pred = df['prediction'].to_numpy().astype(int)
        y_test = df['class'].to_numpy().astype(int)

        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred, normalize='true'))






