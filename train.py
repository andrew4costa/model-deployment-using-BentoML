import xgboost as xgb
import pandas as pd
import bentoml
import warnings

def train_xgb_save(X, y, tag_name="xgb_final"):
    dtrain = xgb.DMatrix(X.values, label=y.values)

    # set parameters
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
    }

    booster = xgb.train(params=params, dtrain=dtrain)

    bento_xgb = bentoml.xgboost.save_model("xgb_booster", booster)

if __name__ == "__main__":
    data = pd.read_csv("/Users/andrewcosta/Desktop/API/data.csv")
    X, y = data.drop("target", axis=1), data[["target"]]

    # Train and save
    train_xgb_save(X, y, "xgb_booster")