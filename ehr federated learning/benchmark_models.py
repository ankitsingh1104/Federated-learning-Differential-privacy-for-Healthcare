from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def run_benchmarks(X_train,X_test,y_train,y_test):

```
results = {}

lr = LogisticRegression(max_iter=1000,class_weight="balanced")
lr.fit(X_train,y_train)

results["LogisticRegression"] = roc_auc_score(
    y_test,
    lr.predict_proba(X_test)[:,1]
)

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample"
)

rf.fit(X_train,y_train)

results["RandomForest"] = roc_auc_score(
    y_test,
    rf.predict_proba(X_test)[:,1]
)

xgb_model = xgb.XGBClassifier(n_estimators=500)

xgb_model.fit(X_train,y_train)

results["XGBoost"] = roc_auc_score(
    y_test,
    xgb_model.predict_proba(X_test)[:,1]
)

return results
```
