import os
import random
import scipy.spatial
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import postprocess
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
def data_process():
    data_path = "data/compas/compas-scores-two-years.csv"
    data = pd.read_csv(data_path, sep=",")
    
    int_values = ["age","juv_fel_count","decile_score","juv_misd_count","juv_other_count","v_decile_score","priors_count"]
    string_values = ["sex","two_year_recid","type_of_assessment","v_type_of_assessment"]
    date_values = ["c_jail_in","c_jail_out","c_offense_date","screening_date","in_custody","out_custody"]
    
    my_attrs = []
    for int_val in int_values:
        my_attrs.append(data[int_val])
    for string_val in string_values:
        my_attrs.append(pd.get_dummies(data[string_val], prefix=string_val, drop_first=True))
    for date_val in date_values:
        temp = pd.to_datetime(data[date_val])
        t_min, t_max = min(temp), max(temp)
        my_attrs.append((temp - t_min) / (t_max - t_min))
    new_data = pd.concat(my_attrs, axis=1)
    new_data["African-American"] = (data["race"] == "African-American")
    new_data = new_data.dropna()
    
    data = new_data.copy()
    data = data.drop(["two_year_recid_1"], axis=1)
    data = data.drop(["African-American"], axis=1)
    
    targets = new_data["two_year_recid_1"].to_numpy().astype(np.float64)
    
    group_names, groups = np.unique(new_data["African-American"], return_inverse=True)
    group_names = ["Others", "African-American"]
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
    plt.savefig("compas_images/kde_plot.png")
    
    return data, targets, groups, group_names

def main():
    n_bins = 18
    epsilons = [np.inf, 10, 5, 1, 0.5, 0.1]
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
        
        model_a = NNmodel(input_size=train_X.shape[1], n_classes=2)
        predictor_a = MLPClassifier(model_a, targets="s", loss_fn=torch.nn.CrossEntropyLoss())
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
            plt.savefig("compas_images/no_post_processing.png")
        
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
            plt.savefig("compas_images/chzhen.png")
        
        # Binning
        postprocessor = postprocess.PrivateHDEFairPostProcessor().fit(
            train_targets, train_groups, alpha=1.0, bound=(0, 1), n_bins=n_bins,
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
                bound=(0, 1), n_bins=n_bins, rng=np.random.default_rng(seed)
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
                plt.savefig("compas_images/private.png")
        
        # svd
        predictor.model = compress(model, torch.Tensor(train_X), torch.Tensor(train_groups), torch.Tensor(train_targets),
                                   c_1=15, c_2=150)
        predictor.fine_tune(train_X, train_targets)
        targets_valid_fair = predictor.predict_proba(valid_X).squeeze()
        valid_results[0, -1, k] = mean_squared_error(valid_targets, targets_valid_fair)
        valid_results[1, -1, k] = ks_dist(targets_valid_fair, valid_groups)
        targets_test_fair = predictor.predict_proba(test_X).squeeze()
        results[0, -1, k] = mean_squared_error(test_targets, targets_test_fair)
        results[1, -1, k] = ks_dist(targets_test_fair, test_groups)
        
        if k == 3:
            df = pd.DataFrame({
                name: pd.Series(targets_test_fair[test_groups == a]) for a, name in enumerate(group_names)
            }).plot.kde(linewidth=2)
            plt.legend(loc="upper left")
            plt.savefig("compas_images/svd_{}_{}.png".format(15, 150))
    
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