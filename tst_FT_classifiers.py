import fasttext
import pandas as pd
import os
import pathlib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    classifiers_folder_path = pathlib.Path("./model/twitter/gridsearch")
    classifiers = []
    classifier_files = [
        f
        for f in os.listdir(classifiers_folder_path)
        if os.path.isfile(os.path.join(classifiers_folder_path, f))
           and f.endswith(".ftz")
    ]

    best_f1 = 0
    best_f1_classifier_name = ""

    df_out = pd.DataFrame(columns = ["params", "data", "preprocessing", "accuracy", "F1 score"])

    for i, name in enumerate(classifier_files):
        print("-"*50)
        print(name)
        model = fasttext.load_model(os.path.join(classifiers_folder_path, name))
        data_type = name.split("_")[0]
        preprocessing_type = name.split("_")[1]

        # FT built in classification
        df = pd.read_pickle(f'data/twitter/preprocessed/multi/{preprocessing_type}/test.p')

        res = [i for i in list(df.columns) if i.startswith('tweet')][0]
        df["prediction"] = df[res].apply(lambda x: model.predict(str(x).replace("\n", " "))[0][0][-1:])
        y_pred = df['prediction'].to_numpy().astype(int)
        y_test = df['class'].to_numpy().astype(int)


        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred, normalize='true'))

        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)

        if (f1 > best_f1):
            best_f1 = f1
            best_f1_classifier_name = name

        params = dict()
        args_obj = model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                #print(f"{hparam} -> {getattr(args_obj, hparam)}")
                params[hparam] = getattr(args_obj, hparam)

        param_string = f'dim -> {params["dim"]}\n' \
                       f'epoch -> {params["epoch"]}\n' \
                       f'lr -> {params["lr"]}\n' \
                       f'wordNgrams -> {params["wordNgrams"]}\n' \
                       f'minn -> {params["minn"]}\n' \
                       f'maxn -> {params["maxn"]}\n' \
                       f'loss -> {params["loss"]}\n'



        df_out.loc[name] = [param_string, data_type, preprocessing_type, accuracy, f1]

    print(f"best F1 score: {best_f1}\n classifier name: {best_f1_classifier_name}")

    df_out.to_excel("model/twitter/results2.xlsx")