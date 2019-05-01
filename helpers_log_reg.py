# Benjamin Li, Jeeheh Oh, Jenna Wiens. 2019-05
# contact: benyli@umich.edu

# implements a regularized logistic regression cross-validation, training, and evaluation pipeline
# USAGE:
# import helpers_log_reg as log_reg
# call function:
# log_reg.do_log_reg(X, y, feature_dict, train_indices, test_indices, C_range, k_best_range, n_random_iters, random_split)

# parameters
# X: nxd feature matrix where n = number of examples and d = number of features
# y: d-length labels vector. The ith row (training example) in X _MUST_ correspond to the ith label in y
# feature_dict: for random splits, can pass in dictionary of features with key=index and value=feature name
# train_indices: for temporal splits. These _MUST_ be the first 80% (or 50%, 60%, 90%, etc.) of the rows in X for
    # the temporal c-v to work correctly. For example, if X is 100x20, train_indices MUST be the first 80 rows
# test_indices, for temporal splits. These _MUST_ be the last 20% (or 50%, 40%, 10%, etc.) of the rows in X
# C_range: list of C regularization hyperparameters for L2 regularization to test
# k_best_range: list of k hyperparamters for chi-square feature selection to test. k_best_range = [X.shape[1]] means no filter feature selection
# n_random_iters. for random splits. Number of random split experiments to run
# random_split: True for random splits, False for (single) temporal splits


import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics


def do_log_reg(X, y, feature_dict, train_indices, test_indices, C_range, k_best_range,
    n_random_iters, random_split):

    k_folds = 5
    metric = 'AUROC'

    print_log_reg_diagnostic(k=k_folds, metric=metric, C_range=C_range, penalty='l2', class_weight='balanced',
        n_random_iters=n_random_iters, random_split=random_split)

    if random_split:
        split = 'random'
    elif not random_split:
        split = 'temporal'

    cv_hyperparameter_grid, train_hyperparameter_grid, test_hyperparameter_grid = k_best_tests(X, y,
        train_indices, test_indices, feature_dict, k_folds, metric, C_range, k_best_range, n_random_iters, split=split)

    return cv_hyperparameter_grid, train_hyperparameter_grid, test_hyperparameter_grid


def do_random_split(X, y, train, test,
    feature_dict, k, metric, C_range, split, n_iters, k_features):
    optimal_c_param_counts = {c_param: 0 for c_param in C_range}
    test_AUROCs = []
    train_AUROCs = []
    clfs = []

    train_proportion = 0.80
    train_test_split = int(len(y)*train_proportion)
    print('Using a random {}{}/{}{} train/test split'.format(train_proportion*100,'%',100-train_proportion*100,'%'))

    for i in range(n_iters):

        # get new train, test indices
        master_indices = np.arange(len(y))
        np.random.shuffle(master_indices)

        train = master_indices[ 0:train_test_split ]
        test = master_indices[ train_test_split: ]


        # find best C param
        optimal_c_param,optimal_clf,optimal_cv_performance,_ = select_regularization_hyperparameter(
            X[train], y[train],
            k=k, metric=metric, C_range=C_range, penalty='l2', class_weight='balanced',
            verbose=False, split='random', k_features=k_features)

        # evaluate using current train, test split and optimal C param
        test_AUROC, train_AUROC, y_actual_pred_on_test, y_actual_pred_on_train = \
            evaluate_log_reg(X[train], y[train], X[test], y[test], optimal_c_param, verbose=False)

        optimal_c_param_counts[optimal_c_param] += 1
        test_AUROCs.append(test_AUROC)
        train_AUROCs.append(train_AUROC)
        clfs.append(optimal_clf)

        print('{} / {} randomized train, test iterations completed. chose optimal c = {}'.format(i+1, n_iters, optimal_c_param))
        print('with test AUROC = {} and train AUROC = {}'.format(test_AUROC, train_AUROC))


    confidence_interval = 95
    lower_index=np.around(n_iters*(100-confidence_interval)/(2*100),decimals=0).astype(int)
    upper_index=min((n_iters-1), (n_iters-lower_index))

    median_index = np.argsort(test_AUROCs)[len(test_AUROCs)//2]

    print('\n')
    return print_log_reg_results_random_split(optimal_c_param_counts, test_AUROCs, train_AUROCs, n_iters,
        confidence_interval, lower_index, upper_index, clfs[median_index], feature_dict, optimal_cv_performance)


def do_empirical_bootstrap(X, y, train, test,
    feature_dict, k, metric, C_range, split, k_features,
    cv_hyperparameter_grid, train_hyperparameter_grid, test_hyperparameter_grid, k_index):
    # find best C param
    optimal_c_param,optimal_clf,optimal_cv_performance,all_cv_performances = select_regularization_hyperparameter(
        X[train], y[train],
        k=k, metric=metric, C_range=C_range, penalty='l2', class_weight='balanced',
        verbose=False, split=split, k_features=k_features)
    cv_hyperparameter_grid[:,k_index] = all_cv_performances

    indices_selected = do_feature_selection_on(X[train],y[train],k_features)
    X_train = X[train]
    X_train = X_train[:,indices_selected]
    X_test = X[test]
    X_test = X_test[:,indices_selected]

    # evaluate using current train, test split and optimal C param
    original_AUROC_test, original_AUROC_train, y_actual_pred_on_test, y_actual_pred_on_train = \
        evaluate_log_reg(X_train, y[train], X_test, y[test], optimal_c_param, verbose=False)

    train_hyperparameter_grid, test_hyperparameter_grid = \
        populate_hyperparameter_grids(train_hyperparameter_grid, test_hyperparameter_grid,
            X_train, y[train], X_test, y[test], C_range, k_index)

    print('classifier training completed. chose optimal c = {} with test AUROC = {}'.format(optimal_c_param, original_AUROC_test))

    n_bootstraps = 1000
    confidence_interval = 95

    train_lower_bound,train_original_AUROC,train_upper_bound, \
        train_lower_AUROC_info,train_upper_AUROC_info = evaluate_empirical_bootstrap(n_bootstraps,
        confidence_interval, original_AUROC_train, y[train], y_actual_pred_on_train, 'train AUROC')

    test_lower_bound,test_original_AUROC,test_upper_bound, \
        test_lower_AUROC_info,test_upper_AUROC_info = evaluate_empirical_bootstrap(n_bootstraps,
        confidence_interval, original_AUROC_test, y[test], y_actual_pred_on_test, 'test AUROC')

    return test_lower_bound,test_original_AUROC,test_upper_bound, \
        train_lower_bound,train_original_AUROC,train_upper_bound, \
        optimal_cv_performance,optimal_c_param, \
        cv_hyperparameter_grid,train_hyperparameter_grid,test_hyperparameter_grid


def evaluate_empirical_bootstrap(n_bootstraps, confidence_interval,
    original_AUROC, y_test, y_pred, title):

    # print('starting empirical bootstrap with {} resamples with length len(y_test) ({}) from y_test and y_pred'.format(n_bootstraps, len(y_test)))

    bootstrap_resample_indices = np.random.choice(len(y_test),(n_bootstraps,len(y_test)),replace=True)

    AUROCs = []
    AUROC_dict = {}
    y_test_low = []
    y_pred_low = []
    y_test_high = []
    y_pred_high = []
    for i in range(n_bootstraps):
        current_indices = bootstrap_resample_indices[i,:]
        current_AUROC = metrics.roc_auc_score(y_test[current_indices], y_pred[current_indices])
        AUROCs.append(current_AUROC)
        AUROC_dict[current_AUROC] = [y_test[current_indices], y_pred[current_indices]]

    AUROCs = np.sort(AUROCs)

    # print('finished empirical bootstrap. bootstrapped AUROCs range from {} to {}\n'.format(min(AUROCs), max(AUROCs)))

    delta_stars = AUROCs - original_AUROC
    lower_index=np.around(n_bootstraps*(100-confidence_interval)/(2*100),decimals=0).astype(int)
    upper_index=min((n_bootstraps-1), (n_bootstraps-lower_index))

    lower_bound = round(original_AUROC+delta_stars[lower_index],5)
    upper_bound = round(original_AUROC+delta_stars[upper_index],5)

    print('empirical bootstrap for {} with {} resamples: {} [{}, {}]'.format(title, n_bootstraps,
        round(original_AUROC,3), round(lower_bound,3), round(upper_bound,3)))

    fake_lower_AUROC = AUROCs[lower_index]
    fake_upper_AUROC = AUROCs[upper_index]

    return lower_bound, original_AUROC, upper_bound, AUROC_dict[fake_lower_AUROC], AUROC_dict[fake_upper_AUROC]


def k_best_tests(X, y, train_indices, test_indices,
    feature_dict, k_folds, metric, C_range, k_best_range, n_random_iters, split='random'):
    test_lower_bounds = []
    test_original_AUROCs = []
    test_upper_bounds = []
    train_lower_bounds = []
    train_original_AUROCs = []
    train_upper_bounds = []
    optimal_cv_performances = []
    optimal_c_params_by_k_features = []

    cv_hyperparameter_grid = np.zeros((len(C_range),len(k_best_range)))
    train_hyperparameter_grid = np.zeros((len(C_range),len(k_best_range)))
    test_hyperparameter_grid = np.zeros((len(C_range),len(k_best_range)))

    for k_index,k_features in enumerate(k_best_range):
        copy_of_X = X[:]
        print('\n----------')
        print('testing {} features'.format(k_features))

        if split == 'random':
            test_lower_bound,test_original_AUROC,test_upper_bound, \
                train_lower_bound,train_original_AUROC,train_upper_bound, \
                optimal_cv_performance,median_clf = \
                do_random_split(copy_of_X, y, train_indices, test_indices,
                    feature_dict, k_folds, metric, C_range, split, n_random_iters, k_features)
            optimal_clf = median_clf

        elif split == 'temporal':
            test_lower_bound,test_original_AUROC,test_upper_bound, \
                train_lower_bound,train_original_AUROC,train_upper_bound, \
                optimal_cv_performance,optimal_c_param, \
                cv_hyperparameter_grid,train_hyperparameter_grid,test_hyperparameter_grid = \
                do_empirical_bootstrap(copy_of_X, y, train_indices, test_indices,
                    feature_dict, k_folds, metric, C_range, split, k_features,
                    cv_hyperparameter_grid, train_hyperparameter_grid, test_hyperparameter_grid, k_index)

        test_lower_bounds.append(test_lower_bound)
        test_original_AUROCs.append(test_original_AUROC)
        test_upper_bounds.append(test_upper_bound)
        train_lower_bounds.append(train_lower_bound)
        train_original_AUROCs.append(train_original_AUROC)
        train_upper_bounds.append(train_upper_bound)
        optimal_cv_performances.append(optimal_cv_performance)
        if split == 'temporal':
            optimal_c_params_by_k_features.append(optimal_c_param)


    make_heatmap(cv_hyperparameter_grid, k_best_range, C_range, 'Mean c-v AUROC for different hyperparamters')
    make_heatmap(train_hyperparameter_grid, k_best_range, C_range, 'Train AUROC for different hyperparamters')
    make_heatmap(test_hyperparameter_grid, k_best_range, C_range, 'Test AUROC for different hyperparamters')


    plt.plot(k_best_range,test_lower_bounds,linestyle='--',color='m',linewidth=1.0)
    plt.plot(k_best_range,test_original_AUROCs,linestyle='-',color='m',linewidth=3.0, label='test AUROC')
    plt.plot(k_best_range,test_upper_bounds,linestyle='--',color='m',linewidth=1.0, label='test AUROC 95% CI')
    plt.plot(k_best_range,train_lower_bounds,linestyle='--',color='g',linewidth=1.0)
    plt.plot(k_best_range,train_original_AUROCs,linestyle='-',color='g',linewidth=3.0, label='train AUROC')
    plt.plot(k_best_range,train_upper_bounds,linestyle='--',color='g',linewidth=1.0, label='train AUROC 95% CI')
    plt.plot(k_best_range,optimal_cv_performances,linestyle='-',color='b',linewidth=3.0, label='optimal c-v AUROC')

    plt.title('AUROC vs. number of features')
    plt.xlabel('number of features')
    plt.ylabel('AUROC')
    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(1.05, 1.05))
    plt.show()

    optimal_k_index = np.argmax(optimal_cv_performances)
    print('optimal k: {}'.format(k_best_range[optimal_k_index]))
    if split == 'temporal':
        print('optimal C: {}'.format(optimal_c_params_by_k_features[optimal_k_index]))
    print('optimal c-v performance: {}'.format(round(optimal_cv_performances[optimal_k_index],3)))
    print('train AUROC: {} [{}, {}]'.format(round(train_original_AUROCs[optimal_k_index],3),
        round(train_lower_bounds[optimal_k_index],3),
        round(train_upper_bounds[optimal_k_index],3)))
    print('test AUROC: {} [{}, {}]'.format(round(test_original_AUROCs[optimal_k_index],3),
        round(test_lower_bounds[optimal_k_index],3),
        round(test_upper_bounds[optimal_k_index],3)))

    return cv_hyperparameter_grid, train_hyperparameter_grid, test_hyperparameter_grid


def select_regularization_hyperparameter(X, y,
    k=5, metric='AUROC', C_range=[], penalty='l2', class_weight='balanced',
    verbose=False, split='random', k_features=100):

    # X is X[train] and y is y[train]

    cv_performances = []
    clfs = []

    for c in C_range:
        clf = LogisticRegression(penalty='l2', class_weight='balanced', C=c)
        average_performance = find_cross_validation_performance(clf, X, y,
            k=5, metric=metric, split=split, k_features=k_features)

        if (verbose):
            print('C: {}, average_performance: {}'.format(c, average_performance))

        cv_performances.append(average_performance)


        copy_of_X = X[:]

        # create and fit selector
        selector = SelectKBest(chi2, k=k_features)
        selector.fit(copy_of_X, y)
        # get indices of columns to keep
        indices_selected = selector.get_support(indices=True)
        # create new feature matrix with only desired columns, or overwrite existing
        transformed_X = copy_of_X[:,indices_selected]

        clf.fit(transformed_X, y)
        clfs.append(clf)

    if verbose:
        plt.figure(figsize=(25, 6))
        plt.bar(np.arange(len(C_range)),cv_performances)
        plt.title('Cross-validation AUROC vs. C regularization parameter')
        plt.xlabel('C hyperparameter')
        plt.ylabel('AUROC')
        plt.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
        plt.xticks(np.arange(len(C_range)),C_range)
        plt.show()

    max_index = np.argmax(cv_performances)
    print('best c: {}. best c-v AUROC: {}'.format(C_range[max_index], cv_performances[max_index]))

    return C_range[max_index], clfs[max_index], cv_performances[max_index], cv_performances


def find_cross_validation_performance(clf, X, y,
    k=5, metric='AUROC', split='random', k_features=100):
    scores = []

    # X is X[train] and y is y[train]

    # split options:
    # note forward_chain and nested are only available as test set-scaled and stratified
    # random, naive_temporal, one_fold_temporal, forward_chain, nested

    if split=='random':
        split = 'random'
    elif split=='temporal':
        split = 'naive_temporal'

    if (split == 'random') or (split == 'forward_chain'):
        if split == 'random':
            # multiple shuffled c-v splits
            cv_splitter = StratifiedKFold(n_splits=k, shuffle=True)

        count = k
        for train, test in cv_splitter.split(X, y):
            # fit clf on the currently defined c-v split (see above)
            indices_selected = do_feature_selection_on(X[train],y[train],k_features)
            X_train = X[train]
            X_train = X_train[:,indices_selected]
            X_test = X[test]
            X_test = X_test[:,indices_selected]

            clf.fit(X_train, y[train])

            y_test = y[test]
            y_actual_pred = clf.decision_function(X_test)
            y_pred = clf.predict(X_test)
            scores.append(calculate_performance_metric(y_test, y_pred, y_actual_pred, metric='AUROC'))


    elif (split == 'naive_temporal') or (split == 'one_fold_temporal'):
        # define 5 folds temporally
        groups = np.zeros(len(y))
        groups[round(len(y)*(0/5)):round(len(y)*(1/5))] = 1
        groups[round(len(y)*(1/5)):round(len(y)*(2/5))] = 2
        groups[round(len(y)*(2/5)):round(len(y)*(3/5))] = 3
        groups[round(len(y)*(3/5)):round(len(y)*(4/5))] = 4
        groups[round(len(y)*(4/5)):] = 5

        # equally sized, temporally defined c-v splits
        count = 1
        logo = LeaveOneGroupOut()
        for train, test in logo.split(X, y, groups=groups):
            # fit clf on the current 80% c-v split
            indices_selected = do_feature_selection_on(X[train],y[train],k_features)
            X_train = X[train]
            X_train = X_train[:,indices_selected]
            X_test = X[test]
            X_test = X_test[:,indices_selected]

            clf.fit(X_train, y[train])

            y_test = y[test]
            y_actual_pred = clf.decision_function(X_test)
            y_pred = clf.predict(X_test)

            if split == 'naive_temporal':
                scores.append(calculate_performance_metric(y_test, y_pred, y_actual_pred, metric='AUROC'))
            count += 1

    # return the average performance across all fold splits
    return np.mean(np.array(scores))


def calculate_performance_metric(y_test, y_pred, y_actual_pred, metric):
    assert(len(y_test) == len(y_pred))
    clf_confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])

    if (metric == 'accuracy'):
        return metrics.accuracy_score(y_test, y_pred)
    elif (metric == 'AUROC'):
        return metrics.roc_auc_score(y_test, y_actual_pred)
    elif (metric == 'precision'):
        return metrics.precision_score(y_test, y_pred)
    elif (metric == 'sensitivity'):
        # sensitivity is the same as recall
        return metrics.recall_score(y_test, y_pred)
    elif (metric == 'specificity'):
        # TN / (TN + FP)
        return (clf_confusion_matrix[0][0]) / \
            (clf_confusion_matrix[0][0] + clf_confusion_matrix[0][1])
    elif (metric == 'ppv'):
        # ppv (positive predictive value) is the same as precision
        # TP / (TP + FP)
        return (clf_confusion_matrix[1][1]) / \
            (clf_confusion_matrix[1][1] + clf_confusion_matrix[0][1])
    elif (metric == 'f1-score'):
        # avoid divide-by-zero error
        if clf_confusion_matrix[0][0] == 0:
            return 0
        else:
            return metrics.f1_score(y_test, y_pred)
    else:
        print('ERROR: def calculate_performance_metric()')


def find_y_pred(y_actual_pred, threshold):
    y_pred = []
    for prediction in y_actual_pred:
        if prediction < threshold:
            y_pred.append(0)
        elif prediction >= threshold:
            y_pred.append(1)

    return np.asarray(y_pred)


def make_confusion_matrix(y_test, y_actual_pred, threshold, verbose=False):
    y_pred = find_y_pred(y_actual_pred, threshold)

    print('sens. = {}'.format(round(calculate_performance_metric(y_test, y_pred, y_actual_pred, 'sensitivity'),4)))
    print('spec. = {}'.format(round(calculate_performance_metric(y_test, y_pred, y_actual_pred, 'specificity'),4)))
    print('ppv = {}'.format(round(calculate_performance_metric(y_test, y_pred, y_actual_pred, 'ppv'),4)))

    if verbose:
        print('clfCMatrix[i][j] is number of observations known to be in i and predicted to be in j')
        print('for example, clfCMatrix[0][0] is TN, clfCMatrix[1][0] is FN')
        clf_confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(clf_confusion_matrix)


def find_most_important_features(clf, feature_dict, num_features_to_track, verbose=False):
    weights = clf.coef_[0]
    lowest_indices = np.argsort(weights)[0:num_features_to_track]
    highest_indices = np.argsort(-weights)[0:num_features_to_track]

    absolute_value_weights = abs(clf.coef_[0])
    lowest_absolute_weights = np.argsort(absolute_value_weights)[0:num_features_to_track]

    low_point_estimates = []
    high_point_estimates = []

    if verbose:
        print('\nfeatures with lowest weights (protective):')
        print('---------')
    count = 1
    for index in lowest_indices:
        if verbose:
            print('{}: [ {} ], weight: {}'.format(count, (feature_dict[index]), round(weights[index],3)))
        low_point_estimates.append( round(weights[index],3) )
        count += 1

    if verbose:
        print('\nfeatures with highest weights (risk):')
        print('---------')
    count = 1
    for index in highest_indices:
        if verbose:
            print('{}: [ {} ], weight: {}'.format(count, (feature_dict[index]), round(weights[index],3)))
        high_point_estimates.append( round(weights[index],3) )
        count += 1

    count = 1
    if verbose:
        print('\nfeatures with lowest absolute weights (least useful in model):')
        print('---------')
    for index in lowest_absolute_weights:
        if verbose:
            print('{}: [ {} ], weight: {}'.format(count, (feature_dict[index]), round(weights[index],3)))
        count += 1

    if verbose:
        print('\n')

    return lowest_indices, low_point_estimates, highest_indices, high_point_estimates


def evaluate_log_reg(X_train, y_train, X_test, y_test, C=1.0, verbose=False):
    clf=LogisticRegression(penalty='l2',class_weight='balanced',C=C)
    clf.fit(X_train,y_train)

    y_pred_on_test = clf.predict(X_test)
    y_actual_pred_on_test = clf.decision_function(X_test)

    y_pred_on_train = clf.predict(X_train)
    y_actual_pred_on_train = clf.decision_function(X_train)

    AUROC_output_test = metrics.roc_auc_score(y_test, y_actual_pred_on_test)
    AUROC_output_train = metrics.roc_auc_score(y_train, y_actual_pred_on_train)

    if (verbose):
        print('You are evaluating a logistic regression model')
        print('----------')
        print('Test AUROC: {}'.format(round(AUROC_output_test,2)))
        make_AUROC_figure(y_test, y_actual_pred_on_test, AUROC_output_test)

        print('Train AUROC: {}'.format(round(AUROC_output_train,2)))
        make_AUROC_figure(y_train, y_actual_pred_on_train, AUROC_output_train)

    return AUROC_output_test, AUROC_output_train, y_actual_pred_on_test, y_actual_pred_on_train


def print_log_reg_diagnostic(k=5, metric='AUROC', C_range=[1, 10], penalty='l2', class_weight='balanced',
    n_random_iters=1, random_split=False, random_indices=False):

    print('You are training a logistic regression')
    print('----------')
    if random_split:
        print('Using {} random train, test iterations (naive bootstrap) with {}-fold cross-validation (with shuffling)'.format(n_random_iters, k))
    elif not random_split:
        print('Using 1 temporal train, test split (empirical bootstrap) with 5-fold temporal cross-validation'.format(k))
    print('Using a C parameter range of {} and evaluating on {}'.format(C_range, metric))
    print('Using {} regularization with {} class weights\n'.format(penalty, class_weight))


def print_log_reg_results_random_split(optimal_c_param_counts, test_AUROCs, train_AUROCs, n_iters,
    confidence_interval, lower_index, upper_index, median_clf, feature_dict, optimal_cv_performance):
    print('Results')
    print('----------\n')

    print('Optimal C parameters chosen')
    for k, v in optimal_c_param_counts.items():
        print(k, v)
    print('\n')

    print('Test AUROC distribution')
    print('----------')
    test_AUROCs = np.sort(test_AUROCs)
    print('0th percentile: {}'.format(test_AUROCs[0]))
    print('25th percentile: {}'.format(test_AUROCs[int(n_iters/4)]))
    print('50th percentile: {}'.format(test_AUROCs[int(n_iters/2)]))
    print('75th percentile: {}'.format(test_AUROCs[int(n_iters/(4/3))]))
    print('100th percentile: {}'.format(test_AUROCs[n_iters-1]))
    print('a pseudo-{}{} confidence interval with the naive bootstrap (with a randomized 80/20 train, test split) is [{}, {}]\n'.format(confidence_interval,'%',
        round(test_AUROCs[lower_index],5), round(test_AUROCs[upper_index],5)))

    print('Train AUROC distribution')
    print('----------')
    train_AUROCs = np.sort(train_AUROCs)
    print('0th percentile: {}'.format(train_AUROCs[0]))
    print('25th percentile: {}'.format(train_AUROCs[int(n_iters/4)]))
    print('50th percentile: {}'.format(train_AUROCs[int(n_iters/2)]))
    print('75th percentile: {}'.format(train_AUROCs[int(n_iters/(4/3))]))
    print('100th percentile: {}'.format(train_AUROCs[n_iters-1]))
    print('a pseudo-{}{} confidence interval with the naive bootstrap (with a randomized 80/20 train, test split) is [{}, {}]\n'.format(confidence_interval,'%',
        round(train_AUROCs[lower_index],5), round(train_AUROCs[upper_index],5)))

    find_most_important_features(median_clf, feature_dict, num_features_to_track=10, verbose=True)

    return round(test_AUROCs[lower_index],5), \
        round(test_AUROCs[int(n_iters/2)],5), \
        round(test_AUROCs[upper_index],5), \
        round(train_AUROCs[lower_index],5), \
        round(train_AUROCs[int(n_iters/2)],5), \
        round(train_AUROCs[upper_index],5), \
        optimal_cv_performance,\
        median_clf


def do_feature_selection_on(X, y, k_features):
    copy_of_X = X[:]

    # create and fit selector
    selector = SelectKBest(chi2, k=k_features)
    selector.fit(copy_of_X, y)
    # get indices of columns to keep
    indices_selected = selector.get_support(indices=True)
    # create new feature matrices with only desired columns, or overwrite existing
    transformed_X = copy_of_X[:,indices_selected]

    return indices_selected


def populate_hyperparameter_grids(train_hyperparameter_grid, test_hyperparameter_grid,
    X_train, y_train, X_test, y_test, C_range, k_index):

    # evaluate using current train, test split and all C params
    for c_index,c in enumerate(C_range):
        AUROC_test, AUROC_train, y_actual_pred_on_test, y_actual_pred_on_train = \
            evaluate_log_reg(X_train, y_train, X_test, y_test, c, verbose=False)

        train_hyperparameter_grid[c_index, k_index] = AUROC_train
        test_hyperparameter_grid[c_index, k_index] = AUROC_test

    return train_hyperparameter_grid, test_hyperparameter_grid


def make_AUROC_figure(y_test, y_actual_pred, AUROC_output, n_bootstraps,
    save=False, name_in='default.png', format_in='png', dpi_in=1000):

    n_bootstraps = n_bootstraps
    confidence_interval = 95
    train_lower_bound,train_original_AUROC,train_upper_bound,lower_AUROC_info,upper_AUROC_info = \
        evaluate_empirical_bootstrap(n_bootstraps, confidence_interval,
        AUROC_output, y_test, y_actual_pred, 'AUROC figure')

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_actual_pred)
    fpr_low, tpr_low, thresholds_low = metrics.roc_curve(lower_AUROC_info[0], lower_AUROC_info[1])
    fpr_high, tpr_high, thresholds_high = metrics.roc_curve(upper_AUROC_info[0], upper_AUROC_info[1])

    plt.figure()
    plt.plot(fpr, tpr, 'r', linewidth=3, label = 'AUROC on held-out set = %0.3f' % AUROC_output)
    plt.plot(fpr_low, tpr_low, 'r--', label = 'AUROC 95% confidence interval')
    plt.plot(fpr_high, tpr_high, 'r--')
    plt.plot([0, 1], [0, 1], 'b-.', label = 'AUROC = 0.500 reference')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if save:
        plt.savefig(name_in, format=format_in, dpi=dpi_in)

    plt.show()


def make_heatmap(grid_in, k_best_range, C_range, title_in,
    save=False, name_in='default.png', format_in='png', dpi_in=1000):

    # credit: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    fig, ax = plt.subplots()
    heatmap = ax.imshow(grid_in)

    # we want to show all ticks
    ax.set_xticks(np.arange(len(k_best_range)))
    ax.set_yticks(np.arange(len(C_range)))
    # and label them with the respective list entries
    ax.set_xticklabels(k_best_range, fontsize=16)
    ax.set_yticklabels(C_range, fontsize=16)
    ax.set_xlabel('k (features remaining) feature selection hyperparameter', fontsize=16)
    ax.set_ylabel('C regularization hyperparameter', fontsize=16)

    # rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
        rotation_mode='anchor')

    # loop over data dimensions and create text annotations
    for i in range(len(C_range)):
        for j in range(len(k_best_range)):
            text = ax.text(j, i, '{:0<4}'.format(round(grid_in[i, j],2)),
                ha='center', va='center', color='k', fontsize=12)

    ax.set_title(title_in, fontsize=18)
    fig.tight_layout()

    fig.set_size_inches(len(k_best_range),len(C_range))

    if save:
        plt.savefig(name_in, format=format_in, dpi=dpi_in)

    plt.show()
