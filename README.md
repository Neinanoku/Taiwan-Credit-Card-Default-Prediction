### **Taiwan Credit-Card Default Prediction**

#### Project setup  
To successfully launch the project, please follow the steps below.

1. **Download the dataset**  
   [Default of Credit Card Clients – UCI](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
   Save the file *default of credit card clients.csv* inside the project’s **`data/`** folder (create it if necessary).

2. **Create and activate a Python environment**  
   <details>
   <summary>Conda (recommended)</summary>

   ```bash
   # create env with all pinned versions
   conda env create -f environment.yml
   conda activate credit-default-ml

### **Code Explanation**

```python
def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return np.sign(x) * np.log1p(np.abs(x))

log_transformer = FunctionTransformer(signed_log1p, validate=False)
```
The feqtures are distributed with a long tail. log operator smooths out the spread.

Wrapped in a FunctionTransformer for easy insertion into a Pipeline.

```python
def make_preprocessor(df):
    bills_and_pays = [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
    num_cols = [
        "LIMIT_BAL", "AGE",
        *[f"PAY_RATIO{i}" for i in range(1, 7)],
        "TOTAL_BILL_6M", "TOTAL_PAY_6M", "LATE_MONTHS_COUNT"
    ]
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    pay_status_cols = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]

    return ColumnTransformer([
        ("log_bill", Pipeline([("log", log_transformer), ("scaler", StandardScaler())]), bills_and_pays),
        ("scale_num", StandardScaler(), num_cols),
        ("onehot_cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ("pass_pay", "passthrough", pay_status_cols)
    ], remainder="drop", verbose_feature_names_out=False)
```
Makes an object, ColumnTransformer, that:
1. log_bill: log(BILL_AMT + PAY_AMT)
2. scale_num: LIMIT_BAL, AGE, PAY_RATIOi, TOTAL_BILL_6M, TOTAL_PAY_6M, LATE_MONTHS_COUNT.

PAY_RATIOi, TOTAL_BILL_6M, TOTAL_PAY_6M and LATE_MONTHS_COUNT are feqtures that I made manually. Explanation later.

4. onehot_cat: SEX, EDUCATION, MARRIAGE (binary features)
5. pass_pay: PAY_0 - PAY_6

```python
def make_pipeline(clf, preprocessor):
    return ImbPipeline([
        ("preproc", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf)
    ])
```
Preprocessing - features are translated into a single numerical space.

SMOTE - generates synthetic objects only from training data (Pipeline prevents leakage).

Classifier - the final model.

```python
df = pd.read_csv(..., sep=';', header=1)
df.drop_duplicates()
df.fillna(0)
```

The dataset contains 1 row with the header "0", so header=1.
Filling the gaps with zeros (there are few of them and only in numeric fields).

```python
    for i in range(1, 7):
        denom = df[f"BILL_AMT{i}"] + 1
        df[f"PAY_RATIO{i}"] = np.where(denom > 0, df[f"PAY_AMT{i}"] / denom, 0.0)
```
Loop over the last 6 months (i = 1 … 6).

BILL_AMT i   = amount still owed at the statement date.
PAY_AMT i   = amount actually paid by the customer next month.

PAY_RATIO i   = relative payment = payment / bill.
+1 is added to the denominator to avoid division by zero.

Stored as six new features PAY_RATIO1 … PAY_RATIO6.
These ratios measure the customer’s payment discipline instead of raw cash values.

```python
    df["TOTAL_BILL_6M"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["TOTAL_PAY_6M"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
```

Row-wise sum of all six statement balances → TOTAL_BILL_6M.

Row-wise sum of all six payments → TOTAL_PAY_6M.

These capture the overall spending and repayment volume over half a year.

```python
    pay_status_cols = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    df["LATE_MONTHS_COUNT"] = (df[pay_status_cols] > 0).sum(axis=1)
```

PAY_0, PAY_2 … PAY_6 hold the delay status (0 = on time, 1…9 = months past due).

For each row, count how many of these statuses are positive (> 0).

Resulting feature LATE_MONTHS_COUNT quantifies how often the client was late during the last six months.

```python
    target_col = "default payment next month"
    X, y = df.drop(columns=["ID", target_col]), df[target_col]
```
Define target and feature matrix

y = 1 if the customer defaulted next month, else 0.

X = all remaining columns except ID and the target.

```python
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
```
75 % training, 25 % hold-out test.

stratify=y preserves the original class ratio (~22 % defaults) in both sets.

random_state fixes the shuffle for reproducibility.


```python
    preprocessor = make_preprocessor(df)
```

Calls the helper defined earlier.

Now we define our models we want to compare:
```python
    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE)
    }

    param_grids = {
        "KNN": {
            "clf__n_neighbors": [1, 3, 5, 7, 10, 15, 20, 30, 40, 50],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        },
        "SVM": {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1]
        },
        "AdaBoost": {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.5, 1.0, 1.5]
        }
    }

    scoring_balacc = 'balanced_accuracy'
```
Fitting the models and finding the best ones:
```python
    best_pipes, results = {}, {}
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\n=== GridSearch ({name}) optimise BalAcc ===")
        grid = GridSearchCV(
            make_pipeline(model, preprocessor),
            param_grids[name],
            cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
            scoring=scoring_balacc,
            n_jobs=-1,
            return_train_score=True
        )
        grid.fit(X_train, y_train)
        print("Best params:", grid.best_params_)
        print("CV BalAcc:", grid.best_score_)

        best = grid.best_estimator_
        best_pipes[name] = best

        # --- metrics ---
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best.named_steps["clf"], "predict_proba") else best.decision_function(X_test)

        res = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "BalAcc": balanced_accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_proba)
        }
        results[name] = res

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['ROC_AUC']:.3f})")

    # -- ROC plot
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves (BalAcc‑optimised)"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("roc_curves_v3.png"); plt.close()

    # -- PR curve
    plt.figure(figsize=(8,6))
    for name, best in best_pipes.items():
        y_pred_final = best.predict(X_test)
        print(f"Confusion matrix ({name}):", confusion_matrix(y_test, y_pred_final))
        print(classification_report(y_test, y_pred_final))
```

### **Results**

```bash
=== GridSearch (KNN) optimise BalAcc ===
Best params: {'clf__n_neighbors': 50, 'clf__p': 1, 'clf__weights': 'uniform'}
CV BalAcc: 0.6816412508324116

=== GridSearch (SVM) optimise BalAcc ===
Best params: {'clf__C': 1, 'clf__gamma': 0.01}
CV BalAcc: 0.7024647740159432

=== GridSearch (AdaBoost) optimise BalAcc ===
Best params: {'clf__learning_rate': 0.5, 'clf__n_estimators': 50}
CV BalAcc: 0.6942668684784111
Confusion matrix (KNN): 
[[4262 1579]
 [ 614 1045]]
              precision    recall  f1-score   support

           0       0.87      0.73      0.80      5841
           1       0.40      0.63      0.49      1659

    accuracy                           0.71      7500
   macro avg       0.64      0.68      0.64      7500
weighted avg       0.77      0.71      0.73      7500

Confusion matrix (SVM): 
[[4841 1000]
 [ 715  944]]
              precision    recall  f1-score   support

           0       0.87      0.83      0.85      5841
           1       0.49      0.57      0.52      1659

    accuracy                           0.77      7500
   macro avg       0.68      0.70      0.69      7500
weighted avg       0.79      0.77      0.78      7500

Confusion matrix (AdaBoost): 
[[5086  755]
 [ 806  853]]
              precision    recall  f1-score   support

           0       0.86      0.87      0.87      5841
           1       0.53      0.51      0.52      1659

    accuracy                           0.79      7500
   macro avg       0.70      0.69      0.69      7500
weighted avg       0.79      0.79      0.79      7500
```

High recall - The model captures most real defaulters, that means very few false-negatives.

Great when the bank prefers to miss none, even if it means extra manual reviews.
_________________________

High precision - Among all clients flagged as defaulters, a large share truly are defaulters.

It measures the purity of positive predictions, not the overall accuracy.
_________________________

KNN: Catches many defaulters (high recall) but raises many false alarms (low precision). Good if the bank fears missing risky clients, but review workload will be high.

SVM: Offers the best blend of recall and specificity, plus decent precision. A solid default choice when you need a balanced error profile.

AdaBoost: Generates fewer false alerts (higher precision) but misses more defaulters (lower recall). Suitable when the cost of a false alarm is high.

_________________________

Let's see the graphs:
![roc_curves_v3](https://github.com/user-attachments/assets/244cb206-4d5d-4540-9d84-59caf3fbcd10)

All three models sit almost on top of each other, so their overall AUC is basically identical.

![pr_curves](https://github.com/user-attachments/assets/022bdd17-dd5e-42b8-bc87-7220697ea5d7)

Here the picture is clearer. AdaBoost has the highest area-under-the-PR curve. That means you can pick a recall target and still keep the best precision with AdaBoost. For the heavily imbalanced default-vs-non-default problem, PR curves give more insight than ROC, and AdaBoost offers the cleanest trade-off between catching defaulters and avoiding false alarms.
