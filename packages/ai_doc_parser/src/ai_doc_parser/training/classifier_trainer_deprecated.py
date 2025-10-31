import re

import joblib
import numpy as np
import pandas as pd
from pdf_inference import functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# import fitz


"""Recalculate Featues"""


def Fraction_Capitalized(x):
    x = str(x)
    # matches = []
    regex = r"\b[A-Z]\w*"
    # for line in x:
    #     matches += re.findall(regex, line)
    matches = re.findall(regex, x)
    frac = len(matches) / len(x.split())
    return round(float(frac), 2)


"""Multiclass Multilable Target Generation"""


def multiclass_multilabel_target_generator(raw_target):
    # 3 labels: is_header, is_first_line, is_toc
    multiclass_multilabel_target = []
    for _, row in raw_target.iterrows():
        target = [
            float(ext) if str(ext) != "nan" and str(ext) != "TABLE_TAG" else ext
            for ext in raw_target.iloc[_].values.tolist()
        ]
        multiclass_multilabel_target.append(target)
    return multiclass_multilabel_target


"""Multiclass Single Target Generation"""


def multiclass_singlelabel_target_generator(raw_target):
    # 5 labels: firstline_header, continuedline_header, firstline_paragraph, continuedline_paragraph, toc
    multiclass_singlelabel_target = []

    for _, row in raw_target.iterrows():
        target_row = raw_target.iloc[_].values.tolist()

        target = [
            (
                float(ext)
                if str(ext) != "nan"
                and str(ext) != "TABLE_TAG"
                and all(val != 1 for val in target_row[3:])
                else ext
            )
            for ext in target_row
        ]

        if target[2] == 1:  # if is TOC
            if target[0] == 1:  # if TOC is header, take it as header or nan
                if target[1] == 1:  # if is firstline
                    multiclass_singlelabel_target.append([1, 0, 0, 0, 0])

                else:
                    multiclass_singlelabel_target.append([0, 1, 0, 0, 0])
            else:  # if TOC is not header, take it as TOC
                multiclass_singlelabel_target.append([0, 0, 0, 0, 1])

        else:
            if target[0] == 1:  # if is header
                if target[1] == 1:  # if is firstline
                    multiclass_singlelabel_target.append([1, 0, 0, 0, 0])

                else:
                    multiclass_singlelabel_target.append([0, 1, 0, 0, 0])

            elif target[0] == 0:  # if NOT header
                if target[1] == 1:  # if is firstline
                    multiclass_singlelabel_target.append([0, 0, 1, 0, 0])

                else:
                    multiclass_singlelabel_target.append([0, 0, 0, 1, 0])

            else:
                multiclass_singlelabel_target.append([np.nan] * len([0, 0, 0, 0, 0]))

    return multiclass_singlelabel_target


"""Choosing Arxiv CFR or Latex"""


def make_version_split(df, df_version):
    new_df = pd.DataFrame()
    for ext in df_version:
        new_df = new_df.append(df[df["version"] == ext])
    return new_df


"""Train Test Split"""


def data_split_multiclass_singlelabel(
    df, run, train_df_version=[], val_df_version=[], removed_columns=[]
):
    Xtrain = Xtest = Ytrain = Ytest = text_test = text_train = version_train = (
        version_test
    ) = page_number_train = page_number_test = np.nan

    if len(train_df_version) > 0 or train_df_version == "ALL_DATA":
        if train_df_version != "ALL_DATA":
            # train_df_version  = df.version.unique()
            # print("df",df.version.unique())
            # print("here")
            train_df = make_version_split(df, train_df_version)
            print(train_df)
            # train_df = train_df.dropna()

        else:
            train_df = df.dropna()

        train_df = train_df.sample(frac=1)

        firstline_header = train_df[train_df["target_str"] == "[1, 0, 0, 0, 0]"]
        continued_header = train_df[train_df["target_str"] == "[0, 1, 0, 0, 0]"]
        firstline_paragraph = train_df[train_df["target_str"] == "[0, 0, 1, 0, 0]"]
        continued_paragraph = train_df[train_df["target_str"] == "[0, 0, 0, 1, 0]"]
        toc = train_df[train_df["target_str"] == "[0, 0, 0, 0, 1]"]
        print()

        X = firstline_header
        y = firstline_header["target"].values.tolist()
        try:
            (
                xtrain_firstline_header,
                xtest_firstline_header,
                ytrain_firstline_header,
                ytest_firstline_header,
            ) = train_test_split(X, y, test_size=0.2)
        except:
            print(
                "ValueError,firstline_header[1, 0, 0, 0, 0]: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters."
            )
            xtrain_firstline_header = xtest_firstline_header = (
                ytrain_firstline_header
            ) = ytest_firstline_header = []

        X = continued_header
        y = continued_header["target"].values.tolist()
        try:
            (
                xtrain_continued_header,
                xtest_continued_header,
                ytrain_continued_header,
                ytest_continued_header,
            ) = train_test_split(X, y, test_size=0.2)
        except:
            print(
                "ValueError,continued_header[0, 1, 0, 0, 0]: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters."
            )
            xtrain_continued_header = xtest_continued_header = (
                ytrain_continued_header
            ) = ytest_continued_header = []

        X = firstline_paragraph
        y = firstline_paragraph["target"].values.tolist()
        try:
            (
                xtrain_firstline_paragraph,
                xtest_firstline_paragraph,
                ytrain_firstline_paragraph,
                ytest_firstline_paragraph,
            ) = train_test_split(X, y, test_size=0.2)
        except:
            print(
                "ValueError,firstline_paragraph[0, 0, 1, 0, 0]: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters."
            )
            xtrain_firstline_paragraph = xtest_firstline_paragraph = (
                ytrain_firstline_paragraph
            ) = ytest_firstline_paragraph = []

        X = continued_paragraph
        y = continued_paragraph["target"].values.tolist()
        try:
            (
                xtrain_continued_paragraph,
                xtest_continued_paragraph,
                ytrain_continued_paragraph,
                ytest_continued_paragraph,
            ) = train_test_split(X, y, test_size=0.2)
        except:
            print(
                "ValueError,continued_paragraph[0, 0, 0, 1, 0]: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters."
            )
            xtrain_continued_paragraph = xtest_continued_paragraph = (
                ytrain_continued_paragraph
            ) = ytest_continued_paragraph = []

        X = toc
        y = toc["target"].values.tolist()
        try:
            xtrain_toc, xtest_toc, ytrain_toc, ytest_toc = train_test_split(
                X, y, test_size=0.2
            )
        except:
            print(
                "ValueError,toc[0, 0, 0, 0, 1]: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters."
            )
            xtrain_toc = xtest_toc = ytrain_toc = ytest_toc = []

        ytrain = (
            ytrain_firstline_header
            + ytrain_continued_header
            + ytrain_firstline_paragraph
            + ytrain_continued_paragraph
            + ytrain_toc
        )
        ytest = (
            ytest_firstline_header
            + ytest_continued_header
            + ytest_firstline_paragraph
            + ytest_continued_paragraph
            + ytest_toc
        )

        xtrain_firstline_header = xtrain_firstline_header.append(
            xtrain_continued_header
        )
        xtrain_firstline_header = xtrain_firstline_header.append(
            xtrain_firstline_paragraph
        )
        xtrain_firstline_header = xtrain_firstline_header.append(
            xtrain_continued_paragraph
        )
        xtrain_firstline_header = xtrain_firstline_header.append(xtrain_toc)

        xtest_firstline_header = xtest_firstline_header.append(xtest_continued_header)
        xtest_firstline_header = xtest_firstline_header.append(
            xtest_firstline_paragraph
        )
        xtest_firstline_header = xtest_firstline_header.append(
            xtest_continued_paragraph
        )
        xtest_firstline_header = xtest_firstline_header.append(xtest_toc)

        xtrain = xtrain_firstline_header
        xtest = xtest_firstline_header

        xtrain["target"] = ytrain
        xtest["target"] = ytest

        xtrain = xtrain.sample(
            frac=1
        )  # need to make change if we do not want to reshuffle
        xtest = xtest.sample(frac=1)  # Here need to make changes if you want

        text_train = xtrain["line"].values.tolist()
        version_train = xtrain["version"].values.tolist()
        page_number_train = xtrain["page_number"].values.tolist()
        Ytrain = [np.argmax(a) for a in xtrain["target"].values.tolist()]
        Xtrain = functions.drop_extra_columns(xtrain, removed_columns)

        text_test = xtest["line"].values.tolist()
        version_test = xtest["version"].values.tolist()
        page_number_test = xtest["page_number"].values.tolist()
        Ytest = [np.argmax(a) for a in xtest["target"].values.tolist()]
        Xtest = functions.drop_extra_columns(xtest, removed_columns)

    val_df = df
    Xval_df = Yval_df = text_val_df = version_val_df = page_number_val_df = np.nan
    if len(val_df_version) > 0 or val_df_version == "ALL_DATA":
        if val_df_version != "ALL_DATA":
            val_df = make_version_split(df, val_df_version)

        if run != "inference":
            val_df = val_df.dropna()

        text_val_df = val_df["line"].values.tolist()
        version_val_df = val_df["version"].values.tolist()
        page_number_val_df = val_df["page_number"].values.tolist()

        try:
            Yval_df = [np.argmax(a) for a in val_df["target"].values.tolist()]
        except:
            Yval_df = np.nan
        Xval_df = functions.drop_extra_columns(val_df, removed_columns)

    return (
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        text_test,
        text_train,
        version_train,
        version_test,
        page_number_train,
        page_number_test,
        Xval_df,
        Yval_df,
        text_val_df,
        version_val_df,
        page_number_val_df,
    )


""" Running Predictions """


def prediction(clf, X):
    predicted = clf.predict(X)
    return predicted


"""Prediction Probabilities"""


def prediction_probability(clf, X):
    predicted = clf.predict_proba(X)
    return predicted


"""F1 Score"""


def FOneScore(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro")
    return f1


"""Accuracy"""


def auc_score(X, Y, clf):
    auc = roc_auc_score(Y, list(clf.predict_proba(X)), multi_class="ovo")
    return auc


"""Confusion Matrix"""


def conf_matrix(Y, predicted):
    conf_matrix = multilabel_confusion_matrix(Y, predicted)
    return conf_matrix


"""Random Forest Classification"""


def random_forest_model(Xtrain, Ytrain, trial=6):
    randomforestclassifier = RandomForestClassifier(
        random_state=1, criterion="entropy"
    )  # ,max_depth=50)

    clf = randomforestclassifier.fit(Xtrain, Ytrain)

    # save the model to disk
    filename = "RandomForestClassifier_modeltest_" + str(trial) + ".sav"
    joblib.dump(clf, filename)

    return clf


"""Main Function for splitting and generating label"""


def call_main(
    df,
    removed_cols,
    train_df_version,
    val_df_version,
    pdf_type,
    mode="multiclass_singlelabel",
    run="train",
):
    df = df.reset_index(drop=True)

    raw_input = df
    raw_input_for_nan = pd.DataFrame()

    if run != "inference":
        if (
            mode == "multiclass_singlelabel"
        ):  # 5 labels: firstline_header, continuedline_header, firstline_paragraph, continuedline_paragraph, toc
            raw_target = df[["is_header", "is_firstline", "is_toc"]]
            raw_input = df.drop(columns=["is_header", "is_firstline", "is_toc"], axis=1)
            target = multiclass_singlelabel_target_generator(raw_target)

        if (
            mode == "multiclass_multilabel"
        ):  # 3 labels: is_header, is_first_line, is_toc
            raw_target = df[["is_header", "is_firstline", "is_toc"]]
            raw_input = df.drop(columns=["is_header", "is_firstline", "is_toc"], axis=1)
            target = multiclass_multilabel_target_generator(raw_target)

        raw_input["target"] = target
        raw_input["target_str"] = [
            str(ext) for ext in raw_input["target"].values.tolist()
        ]

        raw_input["page_number"] = [
            str(ext) for ext in raw_input["page_number"].values.tolist()
        ]

        if pdf_type == "CFR":
            print("here")
            groups_based_on_version = raw_input.groupby("version")
            versions = raw_input["version"].unique()

            # if any nan or istable is detected in a apge, remove indexes after that nan/istable index
            new_df = pd.DataFrame()
            # raw_input_for_nan= pd.DataFrame()
            for version in versions:
                group_version = groups_based_on_version.get_group(
                    str(version)
                ).reset_index()
                groups = group_version.groupby("page_number")
                pages = group_version["page_number"].unique()
                for page in pages:
                    if str(page) == "nan":
                        continue
                    group = groups.get_group(str(page)).reset_index(drop=True)
                    records = [
                        _
                        for _, ext in enumerate(group["target_str"].values.tolist())
                        if str(ext) == "[nan, nan, nan, nan, nan]"
                    ]
                    pass_flag = False

                    if len(records) == 1:  # if first line of a a page is nan
                        if records[0] == 0:
                            pass_flag = True

                    toc_indices = [
                        _
                        for _, ext in enumerate(group["target"].values.tolist())
                        if ext[4] == 1
                    ]  # if any value in page is is_toc
                    if any(toc_indices):  # if any value in page is is_toc
                        pass_flag = True
                        group = group.iloc[toc_indices]

                    if len(records) == 0 or pass_flag:
                        new_df = new_df.append(group)
                    else:
                        new_df = new_df.append(group)
            raw_input = new_df[new_df["target_str"] != "[nan, nan, nan, nan, nan]"]

    (
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        text_test,
        text_train,
        version_train,
        version_test,
        page_number_train,
        page_number_test,
        Xval_df,
        Yval_df,
        text_val_df,
        version_val_df,
        page_number_val_df,
    ) = data_split_multiclass_singlelabel(
        raw_input, run, train_df_version, val_df_version, removed_cols
    )

    return (
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        text_test,
        text_train,
        version_train,
        version_test,
        page_number_train,
        page_number_test,
        Xval_df,
        Yval_df,
        text_val_df,
        version_val_df,
        page_number_val_df,
        raw_input_for_nan,
    )


"""Training and Testing Model"""


def train_and_test(
    df_list,
    removed_cols_list,
    train_df_version_list,
    val_df_versionn_list,
    pdf_typen_list,
    mode_list,
    run_list,
):
    print("here")

    # Xtrain = Xtest =  Xval_df = pd.DataFrame()
    # Ytrain = Ytest = text_test = text_train = version_train = version_test = page_number_train = page_number_test = Yval_df = text_val_df = version_val_df = page_number_val_df = raw_input_for_nan = []

    # for _,element in enumerate(df_list):
    #       Xtrain_ , Xtest_ , Ytrain_ , Ytest_ , text_test_ , text_train_, version_train_, version_test_, page_number_train_, page_number_test_, Xval_df_ , Yval_df_ , text_val_df_, version_val_df_, page_number_val_df_,raw_input_for_nan_ = call_main(df_list[_],removed_cols_list[_],train_df_version_list[_],val_df_version_list[_],pdf_type_list[_],mode_list[_],run_list[_])
    #       Xtrain = Xtrain.append(Xtrain_)
    #       Xtest = Xtest.append(Xtest_)
    #       Ytrain = Ytrain+Ytrain_
    #       Ytest = Ytest+Ytest_
    #       text_test = text_test+text_test_
    #       text_train = text_train+text_train_
    #       version_train = version_train+version_train_
    #       version_test = version_test+version_test_
    #       page_number_train = page_number_train+page_number_train_
    #       page_number_test = page_number_test+page_number_test_
    #       Xval_df = Xval_df.append(Xval_df_)
    #       Yval_df = Yval_df+Yval_df_
    #       text_val_df = text_val_df+text_val_df_
    #       version_val_df = version_val_df+version_val_df_
    #       page_number_val_df = page_number_val_df+page_number_val_df_
    (
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        text_test,
        text_train,
        version_train,
        version_test,
        page_number_train,
        page_number_test,
        Xval_df,
        Yval_df,
        text_val_df,
        version_val_df,
        page_number_val_df,
        raw_input_for_nan,
    ) = call_main(
        df_list,
        removed_cols_list,
        train_df_version_list,
        val_df_versionn_list,
        pdf_typen_list,
        mode_list,
        run_list,
    )

    print("train starts...")

    is_table = Xtrain["is_table"].values.tolist()
    Xtrain = Xtrain.drop(["is_table"], axis=1)
    Xtrain = Xtrain.drop(["index"], axis=1)
    Xtrain = Xtrain.drop(["target"], axis=1)
    Xtrain = Xtrain.drop(["target_str"], axis=1)
    Xtrain = Xtrain.fillna(0)

    print(Xtrain.columns)

    clf = random_forest_model(Xtrain, Ytrain)

    print("train is Done!")

    y_pred = prediction(clf, Xtrain)
    f1 = FOneScore(Ytrain, y_pred)

    print("\nf1Score on train set is: ", f1)
    print("Confusion_Matrix on train set is: ")
    print(conf_matrix(Ytrain, y_pred))

    Xtrain["line"] = text_train
    Xtrain["target"] = Ytrain
    Xtrain["pred"] = y_pred
    Xtrain["version"] = version_train
    Xtrain["page_number"] = page_number_train
    Xtrain["is_table"] = is_table

    is_table = Xtest["is_table"].values.tolist()
    Xtest = Xtest.drop(["is_table"], axis=1)
    Xtest = Xtest.drop(["index"], axis=1)
    Xtest = Xtest.drop(["target"], axis=1)
    Xtest = Xtest.drop(["target_str"], axis=1)
    Xtest = Xtest.fillna(0)
    print(Xtest)

    y_pred = prediction(clf, Xtest)
    f1 = FOneScore(Ytest, y_pred)

    print("\nf1Score on test set is: ", f1)
    print("Confusion_Matrix on test set is: ")
    print(conf_matrix(Ytest, y_pred))

    Xtest["line"] = text_test
    Xtest["target"] = Ytest
    Xtest["pred"] = y_pred
    Xtest["version"] = version_test
    Xtest["page_number"] = page_number_test
    Xtest["is_table"] = is_table

    return Xtrain, Xtest


"""Valiadation Model"""


def validation(
    df, removed_cols, train_df_version, val_df_version, model, pdf_type, mode, run
):
    (
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        text_test,
        text_train,
        version_train,
        version_test,
        page_number_train,
        page_number_test,
        Xval_df,
        Yval_df,
        text_val_df,
        version_val_df,
        page_number_val_df,
        raw_input_for_nan,
    ) = call_main(
        df, removed_cols, train_df_version, val_df_version, pdf_type, mode, run
    )

    clf = model

    is_table = Xval_df["is_table"].values.tolist()
    Xval_df = Xval_df.drop(["is_table"], axis=1)

    val_pred = prediction(clf, Xval_df)
    f1 = FOneScore(Yval_df, val_pred)

    print("\nf1Score on val set is: ", f1)
    print("Confusion_Matrix on val set is: ")
    print(conf_matrix(Yval_df, val_pred))

    Xval_df["text"] = text_val_df
    Xval_df["target"] = Yval_df
    Xval_df["pred"] = val_pred
    Xval_df["version"] = version_val_df
    Xval_df["page_number"] = page_number_val_df
    Xval_df["is_table"] = is_table

    Xval_df_for_nan = raw_input_for_nan
    if str(raw_input_for_nan) != "nan" and len(raw_input_for_nan) != 0:
        (
            Xtrain_for_nan,
            Xtest_for_nan,
            Ytrain_for_nan,
            Ytest_for_nan,
            text_test_for_nan,
            text_train_for_nan,
            version_train_for_nan,
            version_test_for_nan,
            page_number_train_for_nan,
            page_number_test_for_nan,
            Xval_df_for_nan,
            Yval_df_for_nan,
            text_val_df_for_nan,
            version_val_df_for_nan,
            page_number_val_df_for_nan,
        ) = data_split_multiclass_singlelabel(
            raw_input_for_nan, run, train_df_version, val_df_version, removed_cols
        )

        is_table = Xval_df_for_nan["is_table"].values.tolist()
        Xval_df_for_nan = Xval_df_for_nan.drop(["is_table"], axis=1)

        val_pred_for_nan = prediction(clf, Xval_df_for_nan)
        f1 = FOneScore(Yval_df_for_nan, val_pred_for_nan)
        confmatrix = conf_matrix(Yval_df_for_nan, val_pred_for_nan)

        # val_probs_for_nan = prediction_probability(clf,Xval_df_for_nan)

        Xval_df_for_nan["text"] = text_val_df_for_nan
        Xval_df_for_nan["target"] = Yval_df_for_nan
        Xval_df_for_nan["pred"] = val_pred_for_nan
        Xval_df_for_nan["version"] = version_val_df_for_nan
        Xval_df_for_nan["page_number"] = page_number_val_df_for_nan
        Xval_df_for_nan["is_table"] = is_table

        # Xval_df_for_nan_for_computation = Xval_df_for_nan

        # y = Xval_df_for_nan_for_computation["target"].values.tolist()
        # x = Xval_df_for_nan_for_computation.drop(["text","version","page_number","pred","target","is_table"],axis=1)
        # val_pred = prediction(clf,x)
        # f1 = FOneScore(y,val_pred)
        # confmatrix = conf_matrix(y,val_pred)

        print("\nf1Score on val set is: ", f1)
        print("Confusion_Matrix on val set is: ")
        print(confmatrix)

    a = Xval_df
    b = Xval_df_for_nan

    a = a.append(b)

    a = a.dropna()

    y = a["target"].values.tolist()
    x = a.drop(["text", "version", "page_number", "pred", "target", "is_table"], axis=1)

    val_pred = prediction(clf, x)
    f1 = FOneScore(y, val_pred)

    print("\nOVERALL f1Score on val set is: ", f1)
    print("OVERALL Confusion_Matrix on val set is: ")
    print(conf_matrix(y, val_pred))

    return Xval_df, Xval_df_for_nan


"""Inference Model"""


def inference(
    df, removed_cols, train_df_version, val_df_version, model, pdf_type, mode, run
):
    (
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        text_test,
        text_train,
        version_train,
        version_test,
        page_number_train,
        page_number_test,
        Xval_df,
        Yval_df,
        text_val_df,
        version_val_df,
        page_number_val_df,
        raw_input_for_nan,
    ) = call_main(
        df, removed_cols, train_df_version, val_df_version, pdf_type, mode, run
    )

    Xval_df["text"] = text_val_df
    Xval_df["version"] = version_val_df
    Xval_df["page_number"] = page_number_val_df

    Xval_df = Xval_df.dropna()

    Xval_df["pred"] = [np.nan] * len(Xval_df)

    Xval_df_with_table = Xval_df[Xval_df["is_table"] == True]
    Xval_df_NO_table = Xval_df[Xval_df["is_table"] == False]

    text_val_df = Xval_df_NO_table["text"].values.tolist()
    version_val_df = Xval_df_NO_table["version"].values.tolist()
    page_number_val_df = Xval_df_NO_table["page_number"].values.tolist()
    is_table = Xval_df_NO_table["is_table"].values.tolist()
    Xval_df_NO_table = Xval_df_NO_table.drop(
        ["text", "version", "page_number", "is_table", "pred"], axis=1
    )

    clf = model

    val_pred = prediction(clf, Xval_df_NO_table)

    Xval_df_NO_table["text"] = text_val_df
    Xval_df_NO_table["version"] = version_val_df
    Xval_df_NO_table["page_number"] = page_number_val_df
    Xval_df_NO_table["pred"] = val_pred
    Xval_df_NO_table["is_table"] = is_table

    Xval_df_NO_table = Xval_df_NO_table.append(Xval_df_with_table).sort_index()

    return Xval_df_NO_table


def shifting_lines_with_pages(x):
    df = pd.DataFrame()
    prev_col_list = []
    for i in list(x.columns):
        if i.startswith("prev"):
            prev_col_list.append(i)
    prev_col_list = prev_col_list
    corrent_col_list = [i.replace("prev_line_", "") for i in prev_col_list]
    next_line_col_list = [i.replace("prev_line_", "Next_line_") for i in prev_col_list]
    pages = list(x.Page_No.unique())
    lst = []
    for i in pages:
        # print("pageNo",i)
        temp = x[x.Page_No == i].reset_index()
        header = temp.loc[0]["Class"]
        footer = temp.iloc[-1]["Class"]
        if str(header) != "nan":
            if int(header) != 6:
                # print("changed")
                temp.loc[0, prev_col_list] = list(temp.loc[0, corrent_col_list])
        elif str(footer) != "nan":
            if int(footer) != 7:
                # print("changed")
                temp.loc[temp.index[-1], next_line_col_list] = list(
                    temp.loc[temp.index[-1], corrent_col_list]
                )
        # df.append(temp,ignore_index=True)
        lst.append(temp)
    df = pd.concat(lst, ignore_index=True)
    return df


def shifting(x, df):
    prev = "prev_line" + "_" + x
    next_ = "Next_line" + "_" + x

    df[prev] = df[x].shift(1)
    df[next_] = df[x].shift(-1)
    df.iloc[0, df.columns.get_loc(prev)] = df.iloc[0][x]
    df.iloc[-1, df.columns.get_loc(next_)] = df.iloc[-1][x]

    df = shifting_lines_with_pages(df)

    return df


# df = pd.DataFrame()
# path = '/home/dshah/Inspird-2023-dev/Training_Dataset/pdf_label_file_for_training'
# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith("pdflabeledbyxml.csv"):
#             df = df.append(pd.read_csv(os.path.join(subdir, file)))
# df = pd.read_csv("/home/dshah/Inspird-2023-dev/pdf_parsing_new_version/Full_Trainig_Data_03142023.csv")

# df['No_of_words'] = df.line.apply(lambda x: len(str(x).split()))
# df['fraction_capitalized'] = df.line.apply(lambda x:Fraction_Capitalized(x))

# def convert_bool(x):
#     if x == False:
#         return 0
#     elif x == True:
#         return 1
#     return x

# df['Is_Bold']= df.Is_Bold.apply(lambda x: convert_bool(x))
# df['Is_Italic']= df.Is_Italic.apply(lambda x: convert_bool(x))

# removed_cols = ['Unnamed: 0','line','ID','Class','new_block_order','ncols','table_coordinates','page_number','next_line','version','Page_No','line', 'block_x0', 'block_y0', 'block_x1', 'block_y1', 'line_x0',
#        'line_y0', 'line_x1', 'line_y1','fontColor', 'bgcolor','New_Page_No','Block_ID']

# all_verison  = list(df.version.unique())

# all_verison = [i for i in all_verison if str(i)!= 'nan']

# import random
# random.shuffle(all_verison)
# train_df_version = all_verison[:int((len(all_verison)+1)*.80)] [:150]#Remaining 80% to training set
# val_df_version = all_verison[int((len(all_verison)+1)*.80):][:20]#Splits 20% data to test se

# df = df.reset_index(drop=True)

# df = make_version_split(df,all_verison)


# colms_tobe_added = ['fontSize', 'Is_Bold', 'Is_Italic', 'leftIndent', 'rightspace',
#        'nextLinespace', 'first_char_isdigit', 'first_character_is_special',
#        'first_character_is_Upper', 'first_word_iscompound',
#        'first_word_is_special', 'number_dots', 'No_of_words',
#        'fraction_capitalized']


# for i in colms_tobe_added:
#     shifting(i,df)

# #df['is_page_different'] = df['Page_No']== df['Page_No'].shift(-1)
# df['next_line_page_different'] = df['Page_No']== df['Page_No'].shift(-1)
# df['prev_line_page_different'] = df['Page_No']== df['Page_No'].shift(1)
# df = df.drop(['Next_line_nextLinespace'],axis=1)

# df['page_number'] = df['New_Page_No']

# df['is_table'] = [False for i in range(len(df))]


# cfr_pdf_type = "CFR"
# #arxiv_pdf_type = "Arxiv"

# mode="multiclass_singlelabel"
# run="train"
# Xtrain , Xtest = train_ANd_test(df,removed_cols,train_df_version,val_df_version,cfr_pdf_type,mode_list="multiclass_singlelabel",run_list="train")


# df = df[df['is_table'] == False]
# df = df[(df['is_header'].notna()) | (df['is_toc'].notna()) |(df['is_firstline'].notna())]
# df["is_header"] = df["is_header"].fillna(0)
# df["is_toc"] = df["is_toc"].fillna(0)
# df["is_firstline"] = df["is_firstline"].fillna(0)
# df['No_of_words'] = df.line.apply(lambda x: len(str(x).split()))
# df['fraction_capitalized'] = df.line.apply(lambda x:Fraction_Capitalized(x))
# df['Is_Bold']= df.Is_Bold.apply(lambda x: convert_bool(x))
# df['Is_Italic']= df.Is_Italic.apply(lambda x: convert_bool(x))


# removed_cols = ['Unnamed: 0','line','Class','new_block_order','ncols','table_coordinates','page_number','version','next_line','Page_No']

# all_verison  = list(df.version.unique())

# import random
# random.shuffle(all_verison)
# train_df_version = all_verison[:int((len(all_verison)+1)*.80)]#Remaining 80% to training set
# val_df_version = all_verison[int((len(all_verison)+1)*.80):] #Splits 20% data to test se

# #df = df.head(10000)


# cfr_pdf_type = "CFR"
# #arxiv_pdf_type = "Arxiv"


# mode="multiclass_singlelabel"
# run="train"
# Xtrain , Xtest = train_ANd_test(df,removed_cols,train_df_version,val_df_version,cfr_pdf_type,mode_list="multiclass_singlelabel",run_list="train")
