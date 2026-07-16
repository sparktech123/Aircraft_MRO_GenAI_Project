{
  "best_model": "Logistic Regression",
  "cat_cols": [
    "AircraftModel",
    "PartName",
    "NatureOfConditionA",
    "StageOfOperationCode"
  ],
  "num_cols": [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
  ],
  "text_col": "Discrepancy",
  "feature_cols": [
    "AircraftModel",
    "PartName",
    "NatureOfConditionA",
    "StageOfOperationCode",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "Discrepancy"
  ],
  "trained_rows_final_fit": 277191,
  "trained_date_range": [
    "2021-01-01 00:00:00+00:00",
    "2025-12-31 00:00:00+00:00"
  ],
  "holdout_metrics": {
    "Logistic Regression": {
      "accuracy": 0.7786035101643247,
      "precision": 0.9775303592617839,
      "recall": 0.772788056603397,
      "f1": 0.8631844123417157,
      "roc_auc": 0.8877962839578953
    },
    "Random Forest": {
      "accuracy": 0.7089413589711214,
      "precision": 0.9751559745964244,
      "recall": 0.6956669261321677,
      "f1": 0.8120355054399739,
      "roc_auc": 0.8409274402227662
    },
    "XGBoost": {
      "accuracy": 0.7679972582478038,
      "precision": 0.9847952302845687,
      "recall": 0.7549448136838114,
      "f1": 0.8546863702096891,
      "roc_auc": 0.9016770972785568
    }
  },
  "cv_f1_mean": {
    "Logistic Regression": 0.8645222931016232,
    "Random Forest": 0.8170436583644278,
    "XGBoost": 0.8561414001296924
  },
  "label_rule": {
    "critical_conditions": [
      "BROKEN",
      "CHAFED",
      "CONTAMINATED",
      "CORRODED",
      "CRACKED",
      "DAMAGED",
      "DEFECTIVE",
      "DELAMINATED",
      "DENTED",
      "DETACHED",
      "DISCHARGED",
      "DISCONNECTED",
      "EXCESS PLAY",
      "FAILED",
      "FALSE ACTIVATION",
      "FAULTED",
      "FAULTY",
      "GOUGED",
      "ILLUMINATED",
      "INOPERATIVE",
      "LACK OF LUBE",
      "LEAKING",
      "LOOSE",
      "LOW PRESSURE",
      "MALFUNCTIONED",
      "MISSING",
      "ODOR",
      "OUT OF ADJUST",
      "OUT OF POSITION",
      "PULLED",
      "PUNCTURED",
      "RUSTED",
      "UNSECURE",
      "WORN"
    ],
    "non_event_conditions": [
      "NO DISCREPANCY",
      "NO TEST",
      "NONE",
      "NOT REPORTED",
      "UNKNOWN"
    ],
    "text_fallback_keywords": [
      "failure",
      "fire",
      "smoke",
      "leak",
      "crack",
      "engine issue"
    ]
  }
}