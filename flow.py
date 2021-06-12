from scipy.stats import ttest_ind, wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA
import umap
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)




# train and targets - apart
def flow(train_data, targets, dim_num=2, cv=5, random_state=228, only_one=False) -> pd.DataFrame:
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.to_numpy()

    results = pd.DataFrame(columns=['Reduction_method', 'Classificator', 'Unreduced_acc',
                                    'Reduced_acc', 'Clf_unreducted_time', 'Clf_reducted_time', 'Reduction_time',
                                    'TT_reduction_stat', 'TT_reduction_p', 'TT_time_stat', 'TT_time_p',
                                    'Wilcoxon_reduction_stat', 'Wilcoxon_reduction_p', 'Wilcoxon_time_stat', 'Wilcoxon_time_p'])

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    reduction_names = ['UMAP', 'PCA', 'KPCA', 'LDA']
    reductions = [
        umap.UMAP(n_components=dim_num, random_state=random_state),
        PCA(n_components=dim_num),
        KernelPCA(n_components=dim_num, kernel='rbf'),
        LinearDiscriminantAnalysis(n_components=dim_num),
        #FactorAnalysis(n_components=dim_num, random_state=71)
    ]

    classifieres_names = ['RBF SVM', 'Nearest Neighbors',
                          'Naive Bayes', 'Random Forest']
    #classifieres_names.append('Neural Network')
    classifieres = [
        SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=1e3),
        KNeighborsClassifier(3),
        GaussianNB(),
        RandomForestClassifier(n_estimators=10, max_features=1, random_state=random_state)]

    #classifieres.append(MLPClassifier(alpha=1, max_iter=1000))
    i = 0
    for r_name, r_method in zip(reduction_names, reductions):

        for clf_name, clf in zip(classifieres_names, classifieres):
            # print(f'{r_name} {clf_name}')

            scores_reduced = []
            scores_unreduced = []

            time_reduced = []
            time_unreduced = []
            time_reduction = []
            for train_index, test_index in kf.split(X=train_data, y=targets):
                # cross split
                X_train, X_test = train_data[train_index], train_data[test_index]
                y_train, y_test = targets[train_index], targets[test_index]

                # reduction
                reduction_start_time = time.time()
                if r_name == 'LDA':
                    X_train_reduced = r_method.fit(
                        X_train, y_train).transform(X_train)
                    X_test_reduced = r_method.transform(X_test)
                else:
                    X_train_reduced = r_method.fit_transform(X_train)
                    X_test_reduced = r_method.transform(X_test)
                reduction_finish_time = time.time()
                time_reduction.append(
                    reduction_finish_time-reduction_start_time)

                max_abs = MaxAbsScaler().fit(X_train_reduced)
                X_train_reduced = max_abs.transform(X_train_reduced)
                X_test_reduced = max_abs.transform(X_test_reduced)

                # classification
                # unreduced
                unreduced_start_time = time.time()
                clf.fit(X_train, y_train)
                unreduced_finish_time = time.time()
                scores_unreduced.append(
                    accuracy_score(y_test, clf.predict(X_test)))
                time_unreduced.append(
                    unreduced_finish_time-unreduced_start_time)
                # print(f'unreduced score: {unreduced_score}')

                # reduced
                reduced_start_time = time.time()
                clf.fit(X_train_reduced, y_train)
                reduced_finish_time = time.time()
                scores_reduced.append(accuracy_score(
                    y_test, clf.predict(X_test_reduced)))
                time_reduced.append(reduced_finish_time-reduced_start_time)
                # print(f'reduced score: {reduced_score}')

            # t-Student
            tt_score = ttest_ind(scores_unreduced, scores_reduced)
            tt_time = ttest_ind(time_unreduced, time_reduced)
            # Wilcoxon
            w_score = wilcoxon(
                [x-y for x, y in zip(scores_unreduced, scores_reduced)])
            w_time = wilcoxon(
                [x-y for x, y in zip(time_unreduced, time_reduced)])
            results.loc[i] = [r_name, clf_name, scores_unreduced,
                              scores_reduced, time_unreduced, time_reduced, time_reduction,
                              tt_score[0], tt_score[1], tt_time[0], tt_time[1],
                              w_score[0], w_score[1], w_time[0], w_time[1]]

            i += 1

            if only_one:
                break

        if only_one:
            break
    return results.sort_values(by=['Classificator', 'Reduction_method'])

def flow_for_pair(train_data, targets, r_method, clf, dim_num=2, cv=5, random_state=228):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores_reduced = []
    scores_unreduced = []

    time_reduced = []
    time_unreduced = []
    for train_index, test_index in kf.split(X=train_data, y=targets):
        # cross split
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        # reduction
        reduction_start_time = time.time()
        X_train_reduced = r_method.fit_transform(X_train)
        X_test_reduced = r_method.transform(X_test)
        reduction_finish_time = time.time()

        max_abs = MaxAbsScaler().fit(X_train_reduced)
        X_train_reduced = max_abs.transform(X_train_reduced)
        X_test_reduced = max_abs.transform(X_test_reduced)

        # classification
        # unreduced
        unreduced_start_time = time.time()
        clf.fit(X_train, y_train)
        unreduced_finish_time = time.time()
        scores_unreduced.append(
            accuracy_score(y_test, clf.predict(X_test)))
        time_unreduced.append(
            unreduced_finish_time-unreduced_start_time)
        # print(f'unreduced score: {unreduced_score}')

        # reduced
        reduced_start_time = time.time()
        clf.fit(X_train_reduced, y_train)
        reduced_finish_time = time.time()
        scores_reduced.append(accuracy_score(
            y_test, clf.predict(X_test_reduced)))
        time_reduced.append(reduced_finish_time-reduced_start_time)
    
    return np.mean(scores_reduced)/np.mean(scores_unreduced), np.mean(time_reduced)/np.mean(time_unreduced)