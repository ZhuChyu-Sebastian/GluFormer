from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

seed = 10


def normalize_participant_id(x):
    """Extract participant ID by taking digits 1-7 (skip first digit prefix).

    The data files use different ID prefixes (1xxx vs 2xxx) for the same participants.
    This normalizes them for matching.
    """
    return int(str(int(x))[1:7])

hyperparameter_defaults = dict(
    seed=42,
    measure='bt__hba1c',
    pred_from='GMI',  # 'GMI' or 'Representation'
    perms=1000,

)


# a class to make a dictionary into a dot dict, since we wont use wandb in this demo
# such that you could get items by dot
class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


# make into dot dict
config = DotDict(hyperparameter_defaults)

# print(hyperparameter_defaults)
# wandb.init(config=hyperparameter_defaults, project="name", allow_val_change=True)
# config = wand.config
# print pwd
if config.pred_from == 'GMI':
    # if pred is less than 3 make it 3
    gmi_path = f"Shanghai_GMI.csv"
    gmi_path = gmi_path
    data = pd.read_csv(
        gmi_path,
        index_col=0)
    data = data.select_dtypes(include=[np.number])
    data = data[['GMI']]
elif config.pred_from == 'Representation':
    data = pd.read_csv(
        f"GluFormer_Representations_Shanghai_2023.csv",
        index_col=0)
else:
    raise ValueError("wrong pred_from")

covars = data

# Normalize covars index to participant IDs (strip prefix digit)
covars.index = covars.index.map(normalize_participant_id)
covars = covars.groupby(covars.index).mean()  # Average if duplicate participant IDs

path = "Shanghai_results.csv"
targets_to_predict = pd.read_csv(path, index_col=0)
# set the first col as index
targets_to_predict = targets_to_predict.iloc[:, 0]

# remove rows with nan
targets_to_predict = targets_to_predict[~targets_to_predict.isna()]

# Normalize targets index to participant IDs (strip prefix digit)
targets_to_predict.index = targets_to_predict.index.map(normalize_participant_id)
targets_to_predict = targets_to_predict.groupby(targets_to_predict.index).mean()

# intersection of index between covars and targets_to_predict
ids = covars.index.intersection(targets_to_predict.index)

# if unique ids are less than 100, log to wandb corr of -10000 and exit(0)
unique_ids = len(np.unique(ids))

# wandb.log({"unique_ids": unique_ids})

covars = covars.loc[ids]
targets_to_predict = targets_to_predict.loc[ids]

folds = 5

alpha = 80

print(f"num seeds: {seed}")
corrs_per_seed = []
for seed in range(10):
    name, all_covars, all_targets = 'max', covars, targets_to_predict
    preds, targets = [], []
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(np.unique(ids)):
        X_train, X_test = all_covars.loc[ids[train_index]].values, all_covars.loc[ids[test_index]]
        y_train, y_test = all_targets.loc[ids[train_index]].values, all_targets.loc[ids[test_index]]

        # optional
        y_train = (y_train - y_train.mean()) / y_train.std()

        assert len(set(train_index).intersection(set(test_index))) == 0

        X_test = X_test.groupby(X_test.index).mean().values
        y_test = y_test.groupby(y_test.index).mean().values

        model = Ridge(alpha=alpha)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.append(y_pred)
        targets.append(y_test)

    corr = np.corrcoef(np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze())[0, 1]
    print(f"{config.measure} /  corr: {corr} ")
    corrs_per_seed.append(corr)

corr_mean = np.mean(corrs_per_seed)
stds = np.std(corrs_per_seed)

print(f"best corr: {corr_mean} / best std: {stds}")

random_corrs = []
perms = config.perms
for i in tqdm(range(perms)):
    # permute targets_to_predict index
    targets_to_predict.index = np.random.permutation(targets_to_predict.index)
    name, all_covars, all_targets = 'max', covars, targets_to_predict
    preds = []
    targets = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(np.unique(ids)):
        X_train, X_test = all_covars.loc[ids[train_index]].values, all_covars.loc[ids[test_index]]
        y_train, y_test = all_targets.loc[ids[train_index]].values, all_targets.loc[ids[test_index]]

        y_train = (y_train - y_train.mean()) / y_train.std()

        # asset no intersectio between the train and test indecies
        assert len(set(train_index).intersection(set(test_index))) == 0

        X_test = X_test.groupby(X_test.index).mean().values
        y_test = y_test.groupby(y_test.index).mean().values

        model = Ridge(alpha=alpha)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.append(y_pred)
        targets.append(y_test)
    corr = np.corrcoef(np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze())[0, 1]

    random_corrs.append(corr)
# wandb.log({"number of random corrs above corr": np.sum(np.array(random_corrs) > corr_mean)})

print(f"number of random corrs above corr: {np.sum(np.array(random_corrs) > corr_mean)}")  #
