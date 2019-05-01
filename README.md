## Overview
- This is the code repository for the manuscript ["Using Machine Learning and the Electronic Health Record to Predict Complicated _Clostridium difficile_ Infection"](https://academic.oup.com/ofid/advance-article/doi/10.1093/ofid/ofz186/5475497).
- It includes implementation for a regularized logistic regression cross-validation, training, and evaluation pipeline.

## Usage
1. `import helpers_log_reg as log_reg`
2. Call function: `log_reg.do_log_reg(X, y, feature_dict, train_indices, test_indices, C_range, k_best_range, n_random_iters, random_split)`
    - Input: feature matrix, labels, indices (for temporal data splits), hyperparameter ranges, and settings for random data splits
    - Does: five-fold cross-validation and model-training
    - Output: optimal hyperparameters, model performance, and figures

## Parameters
- **X**: nxd feature matrix where n = number of examples and d = number of features
- **y**: n-length labels vector. The i<sup>th</sup> row (example) in X _must_ correspond to the i<sup>th</sup> label in y
- **feature_dict**: for random splits, can pass in dictionary of features with key=index and value=feature name for analysis of most important features
- **train\_indices**: for temporal splits. These _must_ be the first 80% (or 50%, 60%, 90%, etc.) of the rows in X for the temporal c-v to work correctly. For example, if X is 100x20, train\_indices must be the first 80 rows
- **test\_indices**, for temporal splits. These _must_ be the last 20% (or 50%, 40%, 10%, etc.) of the rows in X
- **C_range**: list of _C_ regularization hyperparameters for L2 regularization to test
- **k\_best\_range**: list of _k_ hyperparamters for chi-square feature selection to test. k\_best\_range = [X.shape[1]] means no filter feature selection
- **n\_random\_iters**. for random splits. Number of random split experiments to run
- **random_split**: True for random splits, False for (single) temporal splits

## Authors
- Benjamin Y. Li, Jeeheh Oh, Vincent B. Young, Krishna Rao, and Jenna Wiens