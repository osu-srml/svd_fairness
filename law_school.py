import os
import random
import scipy.spatial
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import postprocess
import sklearn
import tqdm.auto as tqdm

from models import MLPClassifier, NNmodel
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from svd import compress

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

## Helper functions
def get_convex_hull_lower(x, y):
    
    def get_lower(polygon):
        minx = np.argmin(polygon[:, 0])
        maxx = np.argmax(polygon[:, 0]) + 1
        if minx >= np.argmin(polygon[:, 0]):
            lower_curve = np.concatenate([polygon[minx:], polygon[:maxx]])
        else:
            lower_curve = polygon[minx:maxx]
        return lower_curve
    
    points = np.stack([x, y], axis=1)
    hull = scipy.spatial.ConvexHull(points)
    lower_curve = get_lower(points[hull.vertices])
    return lower_curve

def ks_dist(scores, groups):
    n_groups = len(np.unique(groups))
    max_ks = 0
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            max_ks = max(max_ks, ks_2samp(scores[groups == i], scores[groups == j]).statistic)
    return max_ks


## Preprocess the dataset
def data_transform(df):
    ## Normalize features
    binary_data = pd.get_dummies(df)
    scaler = sklearn.preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(binary_data), columns=binary_data.columns)
    data.index = df.index
    return data

def data_process():
    data_path = "data/law/bar_pass_prediction.csv"
    column_names = ["dnn_bar_pass_prediction", "gender", "lsat",
                    "race1", "pass_bar", "ugpa"]
    #column_names = ["gender", "lsat", "race1", "ugpa"]
    
    original = pd.read_csv(data_path, index_col=0, sep=r",", engine="python", na_values="?")
    #original = original.dropna()
    original["gender"] = original["gender"].astype(str)
    original["race1"] = original["race1"].astype(str)
    original = original[column_names]
    
    #race = ["black", "white"]
    race = ["black", "white"]
    original = original[original["race1"].isin(race)]
    #gender = ["male", "female"]
    #original = original[original["gender"].isin(gender)]
    
    data = original.copy()
    data = data.drop(["ugpa"], axis=1)
    data = data.drop(["race1"], axis=1)
    #data = data.drop(["gender"], axis=1)
    data = data_transform(data)

    targets = original["ugpa"].to_numpy()
    
    group_names, groups = np.unique(original["race1"], return_inverse=True)
    #group_names, groups = np.unique(original["gender"], return_inverse=True)
    group_names = ["African-American", "Others"]
    #group_names = ["male", "female"]
    n_groups = len(group_names)
    
    df = pd.DataFrame(
        zip(np.array(group_names)[groups], targets), columns=["Group", "Target"]
    ).groupby(["Group"]).agg({"count"})
    df.columns = df.columns.droplevel(0)
    df["%"] = df["count"] / df["count"].sum()
    print("data statistics = \n{}".format(df))
    
    df = pd.DataFrame({
        name: pd.Series(targets[groups==a]) for a, name in enumerate(group_names)
    }).plot.kde()
    plt.savefig("law_images/kde_plot.png")
    
    return data, targets, groups, group_names

def main():
    n_bins = 18
    epsilons = [np.inf, 10, 5, 1, 0.5, 0.1]
    #seeds = range(33, 83)
    seeds = range(33, 83)
    split_ratio = 0.3
    
    plt.rcParams.update({"font.size": 14})
    
    valid_results = np.empty((2, 4 + len(epsilons), len(seeds)))
    results = np.empty((2, 4 + len(epsilons), len(seeds)))
    methods = [
        "no postprocessing", "Chzhen et al. (2020)", "binning", "binning + fair"
    ] + [f"binning + private and fair (eps={eps})" for eps in epsilons[1:]] + ["svd"]
    
    data, targets, groups, group_names = data_process()
    
    for k, seed in enumerate(tqdm.tqdm(seeds)):
        seed_everything(seed)
        train_X, test_X, train_targets, test_targets, train_groups, test_groups = train_test_split(
            data, targets, groups, test_size=split_ratio, random_state=seed
        )
        valid_X, test_X, valid_targets, test_targets, valid_groups, test_groups = train_test_split(
            test_X, test_targets, test_groups, test_size=0.5, random_state=seed
        )
        train_X, valid_X, test_X = np.array(train_X).astype(np.float64), np.array(valid_X).astype(np.float64), np.array(test_X).astype(np.float64)
        
        model = NNmodel(input_size=train_X.shape[1])
        predictor = MLPClassifier(model)
        predictor.fit(train_X, train_targets)
        
        from sklearn.linear_model import LogisticRegression
        predictor_a = LogisticRegression()
        predictor_a.fit(train_X, train_groups)
        
        train_scores = predictor.predict_proba(train_X).squeeze()
        valid_scores = predictor.predict_proba(valid_X).squeeze()
        test_scores = predictor.predict_proba(test_X).squeeze()
        valid_group_scores = predictor_a.predict(valid_X).squeeze()
        test_groups_scores = predictor_a.predict(test_X).squeeze()
        
        # No postprocessing
        valid_results[0, 0, k] = mean_squared_error(valid_targets, valid_scores)
        valid_results[1, 0, k] = ks_dist(valid_scores, valid_groups)
        results[0, 0, k] = mean_squared_error(test_targets, test_scores)
        results[1, 0, k] = ks_dist(test_scores, test_groups)
        
        if k == 3:
            df = pd.DataFrame({
                name: pd.Series(test_scores[test_groups == a]) for a, name in enumerate(group_names)
            }).plot.kde(linewidth=2)
            plt.savefig("law_images/no_post_processing.png")
        
        # Chzhen et al. (2020)
        postprocessor = postprocess.WassersteinBarycenterFairPostProcessor().fit(
            train_targets, train_groups, rng=np.random.default_rng(seed)
        )
        targets_valid_fair = postprocessor.predict(valid_scores, valid_group_scores)
        valid_results[0, 1, k] = mean_squared_error(valid_targets, targets_valid_fair)
        valid_results[1, 1, k] = ks_dist(targets_valid_fair, valid_groups)
        targets_test_fair = postprocessor.predict(test_scores, test_groups_scores)
        results[0, 1, k] = mean_squared_error(test_targets, targets_test_fair)
        results[1, 1, k] = ks_dist(targets_test_fair, test_groups)
        
        if k == 3:
            df = pd.DataFrame({
                name: pd.Series(targets_test_fair[test_groups == a]) for a, name in enumerate(group_names)
            }).plot.kde(linewidth=2)
            plt.savefig("law_images/chzhen.png")
        
        # Binning
        postprocessor = postprocess.PrivateHDEFairPostProcessor().fit(
            train_targets, train_groups, alpha=1.0, bound=(1, 4), n_bins=n_bins,
            rng=np.random.default_rng(seed)
        )
        targets_valid_fair = postprocessor.predict(valid_scores, valid_group_scores)
        valid_results[0, 2, k] = mean_squared_error(valid_targets, targets_valid_fair)
        valid_results[1, 2, k] = ks_dist(targets_valid_fair, valid_groups)
        targets_test_fair = postprocessor.predict(test_scores, test_groups_scores)
        results[0, 2, k] = mean_squared_error(test_targets, targets_test_fair)
        results[1, 2, k] = ks_dist(targets_test_fair, test_groups)
        
        # Binning + (private) fair postprocessing
        for i, eps in enumerate(epsilons):
            postprocessor = postprocess.PrivateHDEFairPostProcessor().fit(
                train_targets, train_groups, alpha=0.0, eps=eps,
                bound=(1, 4), n_bins=n_bins, rng=np.random.default_rng(seed)
            )
            targets_valid_fair = postprocessor.predict(valid_scores, valid_group_scores)
            valid_results[0, 3 + i, k] = mean_squared_error(valid_targets, targets_valid_fair)
            valid_results[1, 3 + i, k] = ks_dist(targets_valid_fair, valid_groups)
            targets_test_fair = postprocessor.predict(test_scores, test_groups_scores)
            results[0, 3 + i, k] = mean_squared_error(test_targets, targets_test_fair)
            results[1, 3 + i, k] = ks_dist(targets_test_fair, test_groups)
            
            if k == 3:
                df = pd.DataFrame({
                    name: pd.Series(targets_test_fair[test_groups == a]) for a, name in enumerate(group_names)
                }).plot.kde(linewidth=2)
                plt.savefig("law_images/private.png")
        
        # svd
        predictor.model = compress(model, torch.Tensor(train_X), torch.Tensor(train_groups), torch.Tensor(train_targets),
                                   c_1=15, c_2=150)
        predictor.fine_tune(train_X, train_targets)
        targets_valid_fair = predictor.predict_proba(valid_X).squeeze()
        valid_results[0, -1, k] = mean_squared_error(valid_targets, targets_valid_fair)
        valid_results[1, -1, k] = ks_dist(targets_valid_fair, valid_groups)
        targets_test_fair = predictor.predict_proba(test_X).squeeze()
        results[0, -1, k] = mean_squared_error(targets_test_fair, test_targets)
        results[1, -1, k] = ks_dist(targets_test_fair, test_groups)
        
        if k == 3:
            df = pd.DataFrame({
                name: pd.Series(targets_test_fair[test_groups == a]) for a, name in enumerate(group_names)
            }).plot.kde(linewidth=2)
            plt.legend(loc="upper left")
            plt.savefig("law_images/svd_{}_{}.png".format(15, 150))
            
    means = valid_results.mean(axis=2)
    stds = valid_results.std(axis=2)
    df = pd.DataFrame(np.stack([means[0], stds[0], means[1], stds[1]], axis=1))
    df.columns = ["mse", "mse std", "ks", "ks std"]
    df.index = methods
    
    print("Valid Results = \n{}".format(df))
        
    means = results.mean(axis=2)
    stds = results.std(axis=2)
    df = pd.DataFrame(np.stack([means[0], stds[0], means[1], stds[1]], axis=1))
    df.columns = ["mse", "mse std", "ks", "ks std"]
    df.index = methods
    
    print("Results = \n{}".format(df))

if __name__ == "__main__":
    main()