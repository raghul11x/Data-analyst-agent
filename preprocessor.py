import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Return a fast column-level profile of the dataframe.
    Optimised for speed — samples large datasets.
    """
    rows, cols = df.shape

    sample_df = df if rows <= 5000 else df.sample(5000, random_state=42)

    columns = []
    for col in df.columns:
        series        = df[col]
        sample_series = sample_df[col]

        null_ct  = int(series.isnull().sum())
        null_pct = round(null_ct / rows * 100, 1) if rows > 0 else 0

        unique = int(series.nunique(dropna=True))
        dtype  = str(series.dtype)

        if pd.api.types.is_datetime64_any_dtype(series):
            sem_type = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            sem_type = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            sem_type = "numeric"
        elif unique <= 30 and unique / max(rows, 1) < 0.5:
            sem_type = "categorical"
        else:
            sem_type = "text"

        sample = series.dropna().head(3).astype(str).tolist()

        stats_dict = {}
        if sem_type == "numeric":
            clean = sample_series.dropna()
            if len(clean) > 0:
                try:
                    stats_dict = {
                        "mean": round(float(clean.mean()), 4),
                        "std":  round(float(clean.std()),  4),
                        "min":  round(float(series.min()), 4),
                        "max":  round(float(series.max()), 4),
                        "skew": round(float(clean.skew()), 4),
                    }
                except Exception:
                    stats_dict = {}

        columns.append({
            "name":     col,
            "dtype":    dtype,
            "sem_type": sem_type,
            "null_ct":  null_ct,
            "null_pct": null_pct,
            "unique":   unique,
            "sample":   sample,
            "stats":    stats_dict,
        })

    return {
        "rows":    rows,
        "cols":    cols,
        "columns": columns,
        "memory":  f"{df.memory_usage(deep=False).sum() / 1024:.1f} KB",
    }

def handle_missing(df: pd.DataFrame, rules: list) -> tuple:
    """
    rules: list of {col, strategy}
    strategy: "drop_rows" | "mean" | "median" | "mode" | "ffill" | "bfill" | "constant:<value>"
    Returns (df, log_entries)
    """
    df  = df.copy()
    log = []
    for rule in rules:
        col      = rule["col"]
        strategy = rule["strategy"]
        if col not in df.columns:
            continue
        null_before = int(df[col].isnull().sum())
        if null_before == 0:
            continue

        if strategy == "drop_rows":
            df = df.dropna(subset=[col])
            log.append(f"Dropped {null_before} rows with nulls in '{col}'")
        elif strategy == "mean":
            val = df[col].mean()
            df[col].fillna(val, inplace=True)
            log.append(f"Filled {null_before} nulls in '{col}' with mean ({val:.4f})")
        elif strategy == "median":
            val = df[col].median()
            df[col].fillna(val, inplace=True)
            log.append(f"Filled {null_before} nulls in '{col}' with median ({val:.4f})")
        elif strategy == "mode":
            val = df[col].mode()[0]
            df[col].fillna(val, inplace=True)
            log.append(f"Filled {null_before} nulls in '{col}' with mode ({val})")
        elif strategy == "ffill":
            df[col].fillna(method="ffill", inplace=True)
            log.append(f"Forward-filled {null_before} nulls in '{col}'")
        elif strategy == "bfill":
            df[col].fillna(method="bfill", inplace=True)
            log.append(f"Backward-filled {null_before} nulls in '{col}'")
        elif strategy.startswith("constant:"):
            val = strategy.split(":", 1)[1]
            try:
                val = float(val) if df[col].dtype in [np.float64, np.int64] else val
            except Exception:
                pass
            df[col].fillna(val, inplace=True)
            log.append(f"Filled {null_before} nulls in '{col}' with constant ({val})")

    return df, log

def convert_types(df: pd.DataFrame, conversions: list) -> tuple:
    """
    conversions: list of {col, to_type}
    to_type: "numeric" | "datetime" | "string" | "boolean" | "category"
    """
    df  = df.copy()
    log = []
    for conv in conversions:
        col     = conv["col"]
        to_type = conv["to_type"]
        if col not in df.columns:
            continue
        old_dtype = str(df[col].dtype)
        try:
            if to_type == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif to_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif to_type == "string":
                df[col] = df[col].astype(str)
            elif to_type == "boolean":
                df[col] = df[col].map({"True":True,"False":False,"1":True,"0":False,
                                        "yes":True,"no":False,"Yes":True,"No":False,
                                        1:True, 0:False}).astype("boolean")
            elif to_type == "category":
                df[col] = df[col].astype("category")
            log.append(f"Converted '{col}' from {old_dtype} to {to_type}")
        except Exception as e:
            log.append(f"Failed to convert '{col}' to {to_type}: {e}")
    return df, log

def encode_columns(df: pd.DataFrame, rules: list) -> tuple:
    """
    rules: list of {col, method}
    method: "label" | "onehot" | "target:<target_col>" | "drop"
    """
    df  = df.copy()
    log = []
    for rule in rules:
        col    = rule["col"]
        method = rule["method"]
        if col not in df.columns:
            continue

        if method == "label":
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            log.append(f"Label-encoded '{col}' ({len(uniques)} categories)")

        elif method == "onehot":
            dummies  = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df       = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            log.append(f"One-hot encoded '{col}' → {list(dummies.columns)}")

        elif method.startswith("target:"):
            target_col = method.split(":", 1)[1]
            if target_col in df.columns:
                means    = df.groupby(col)[target_col].mean()
                df[col]  = df[col].map(means)
                log.append(f"Target-encoded '{col}' using mean of '{target_col}'")

        elif method == "drop":
            df.drop(columns=[col], inplace=True)
            log.append(f"Dropped column '{col}'")

    return df, log

def scale_columns(df: pd.DataFrame, rules: list) -> tuple:
    """
    rules: list of {col, method}
    method: "standard" | "minmax" | "robust" | "none"
    """
    df  = df.copy()
    log = []
    for rule in rules:
        col    = rule["col"]
        method = rule["method"]
        if col not in df.columns or method == "none":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            log.append(f"Skipped '{col}'  -  not numeric")
            continue

        series = df[col].dropna()
        if method == "standard":
            mean, std = series.mean(), series.std()
            if std != 0:
                df[col] = (df[col] - mean) / std
                log.append(f"Standard-scaled '{col}' (mean={mean:.3f}, std={std:.3f})")
        elif method == "minmax":
            mn, mx = series.min(), series.max()
            if mx != mn:
                df[col] = (df[col] - mn) / (mx - mn)
                log.append(f"Min-max scaled '{col}' (range {mn:.3f}-{mx:.3f})")
        elif method == "robust":
            median = series.median()
            iqr    = series.quantile(0.75) - series.quantile(0.25)
            if iqr != 0:
                df[col] = (df[col] - median) / iqr
                log.append(f"Robust-scaled '{col}' (median={median:.3f}, IQR={iqr:.3f})")

    return df, log

def handle_outliers(df: pd.DataFrame, rules: list) -> tuple:
    """
    rules: list of {col, method, strategy}
    method:   "iqr" | "zscore"
    strategy: "remove" | "cap" | "keep"
    """
    df  = df.copy()
    log = []
    for rule in rules:
        col      = rule["col"]
        method   = rule.get("method",   "iqr")
        strategy = rule.get("strategy", "cap")
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if method == "iqr":
            q1, q3  = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr     = q3 - q1
            lower   = q1 - 1.5 * iqr
            upper   = q3 + 1.5 * iqr
        else:
            z       = np.abs(stats.zscore(df[col].dropna()))
            lower   = df[col].mean() - 3 * df[col].std()
            upper   = df[col].mean() + 3 * df[col].std()

        mask    = (df[col] < lower) | (df[col] > upper)
        count   = int(mask.sum())
        if count == 0:
            continue

        if strategy == "remove":
            df = df[~mask]
            log.append(f"Removed {count} outliers from '{col}' ({method})")
        elif strategy == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
            log.append(f"Capped {count} outliers in '{col}' to [{lower:.3f}, {upper:.3f}]")
        else:
            log.append(f"Kept {count} outliers in '{col}' (no action)")

    return df, log

def engineer_features(df: pd.DataFrame, rules: list) -> tuple:
    """
    rules: list of {type, ...params}
    Types:
      {type:"datetime_parts", col:"date_col"}
      {type:"log_transform",  col:"col"}
      {type:"sqrt_transform", col:"col"}
      {type:"interaction",    col_a:"a", col_b:"b", op:"multiply"|"divide"|"add"|"subtract"}
      {type:"drop_duplicates"}
      {type:"drop_column", col:"col"}
      {type:"rename", col:"old", new_name:"new"}
      {type:"drop_low_variance", threshold:0.01}
      {type:"drop_high_null", threshold:50}
    """
    df  = df.copy()
    log = []
    for rule in rules:
        t = rule.get("type", "")

        if t == "datetime_parts":
            col = rule["col"]
            if col in df.columns:
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"]    = dt.dt.year
                    df[f"{col}_month"]   = dt.dt.month
                    df[f"{col}_day"]     = dt.dt.day
                    df[f"{col}_weekday"] = dt.dt.dayofweek
                    log.append(f"Extracted year/month/day/weekday from '{col}'")
                except Exception as e:
                    log.append(f"Failed datetime extraction on '{col}': {e}")

        elif t == "log_transform":
            col = rule["col"]
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
                log.append(f"Log-transformed '{col}' → '{col}_log'")

        elif t == "sqrt_transform":
            col = rule["col"]
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
                log.append(f"Sqrt-transformed '{col}' → '{col}_sqrt'")

        elif t == "interaction":
            a, b, op = rule["col_a"], rule["col_b"], rule.get("op", "multiply")
            if a in df.columns and b in df.columns:
                ops = {"multiply": df[a] * df[b], "divide": df[a] / df[b].replace(0, np.nan),
                       "add": df[a] + df[b], "subtract": df[a] - df[b]}
                new_col = f"{a}_{op[:3]}_{b}"
                df[new_col] = ops.get(op, df[a] * df[b])
                log.append(f"Created interaction feature '{new_col}'")

        elif t == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            log.append(f"Dropped {before - len(df)} duplicate rows")

        elif t == "drop_column":
            col = rule["col"]
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                log.append(f"Dropped column '{col}'")

        elif t == "rename":
            col      = rule["col"]
            new_name = rule["new_name"]
            if col in df.columns:
                df.rename(columns={col: new_name}, inplace=True)
                log.append(f"Renamed '{col}' → '{new_name}'")

        elif t == "drop_low_variance":
            thresh  = float(rule.get("threshold", 0.01))
            numeric = df.select_dtypes(include=[np.number])
            variances = numeric.var()
            drop_cols = variances[variances < thresh].index.tolist()
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                log.append(f"Dropped low-variance columns: {drop_cols}")

        elif t == "drop_high_null":
            thresh    = float(rule.get("threshold", 50))
            null_pcts = (df.isnull().sum() / len(df) * 100)
            drop_cols = null_pcts[null_pcts > thresh].index.tolist()
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                log.append(f"Dropped high-null columns (>{thresh}%): {drop_cols}")

    return df, log

def apply_pipeline(df: pd.DataFrame, pipeline: dict) -> tuple:
    """
    pipeline: {
        missing:    [...],
        types:      [...],
        encoding:   [...],
        scaling:    [...],
        outliers:   [...],
        features:   [...],
    }
    Returns (cleaned_df, full_log, preprocessing_code)
    """
    full_log = []

    if pipeline.get("features"):

        pre_rules = [r for r in pipeline["features"]
                     if r.get("type") in ("drop_column", "drop_duplicates",
                                           "drop_high_null", "drop_low_variance", "rename")]
        if pre_rules:
            df, log = engineer_features(df, pre_rules)
            full_log.extend(log)

    if pipeline.get("missing"):
        df, log = handle_missing(df, pipeline["missing"])
        full_log.extend(log)

    if pipeline.get("types"):
        df, log = convert_types(df, pipeline["types"])
        full_log.extend(log)

    if pipeline.get("outliers"):
        df, log = handle_outliers(df, pipeline["outliers"])
        full_log.extend(log)

    if pipeline.get("features"):
        eng_rules = [r for r in pipeline["features"]
                     if r.get("type") not in ("drop_column", "drop_duplicates",
                                               "drop_high_null", "drop_low_variance", "rename")]
        if eng_rules:
            df, log = engineer_features(df, eng_rules)
            full_log.extend(log)

    if pipeline.get("encoding"):
        df, log = encode_columns(df, pipeline["encoding"])
        full_log.extend(log)

    if pipeline.get("scaling"):
        df, log = scale_columns(df, pipeline["scaling"])
        full_log.extend(log)

    code = generate_code(pipeline)
    return df, full_log, code

def generate_code(pipeline: dict) -> str:
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        "df = pd.read_csv('your_dataset.csv')",
        "",
    ]

    if pipeline.get("missing"):
        lines.append("# ── Missing Values ──")
        for r in pipeline["missing"]:
            col, s = r["col"], r["strategy"]
            if s == "drop_rows":
                lines.append(f"df = df.dropna(subset=['{col}'])")
            elif s == "mean":
                lines.append(f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
            elif s == "median":
                lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
            elif s == "mode":
                lines.append(f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")
            elif s == "ffill":
                lines.append(f"df['{col}'].fillna(method='ffill', inplace=True)")
            elif s == "bfill":
                lines.append(f"df['{col}'].fillna(method='bfill', inplace=True)")
            elif s.startswith("constant:"):
                val = s.split(":", 1)[1]
                lines.append(f"df['{col}'].fillna({repr(val)}, inplace=True)")
        lines.append("")

    if pipeline.get("types"):
        lines.append("# ── Type Conversions ──")
        for r in pipeline["types"]:
            col, t = r["col"], r["to_type"]
            if t == "numeric":
                lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
            elif t == "datetime":
                lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
            elif t == "string":
                lines.append(f"df['{col}'] = df['{col}'].astype(str)")
            elif t == "category":
                lines.append(f"df['{col}'] = df['{col}'].astype('category')")
        lines.append("")

    if pipeline.get("outliers"):
        lines.append("# ── Outliers ──")
        for r in pipeline["outliers"]:
            col, m, s = r["col"], r.get("method","iqr"), r.get("strategy","cap")
            if m == "iqr":
                lines.append(f"q1, q3 = df['{col}'].quantile(0.25), df['{col}'].quantile(0.75)")
                lines.append(f"iqr = q3 - q1")
                lines.append(f"lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr")
            else:
                lines.append(f"mean, std = df['{col}'].mean(), df['{col}'].std()")
                lines.append(f"lower, upper = mean - 3*std, mean + 3*std")
            if s == "remove":
                lines.append(f"df = df[(df['{col}'] >= lower) & (df['{col}'] <= upper)]")
            elif s == "cap":
                lines.append(f"df['{col}'] = df['{col}'].clip(lower=lower, upper=upper)")
        lines.append("")

    if pipeline.get("features"):
        lines.append("# ── Feature Engineering ──")
        for r in pipeline["features"]:
            t = r.get("type","")
            if t == "datetime_parts":
                c = r["col"]
                lines.append(f"df['{c}'] = pd.to_datetime(df['{c}'])")
                lines.append(f"df['{c}_year']    = df['{c}'].dt.year")
                lines.append(f"df['{c}_month']   = df['{c}'].dt.month")
                lines.append(f"df['{c}_day']     = df['{c}'].dt.day")
                lines.append(f"df['{c}_weekday'] = df['{c}'].dt.dayofweek")
            elif t == "log_transform":
                c = r["col"]
                lines.append(f"df['{c}_log'] = np.log1p(df['{c}'].clip(lower=0))")
            elif t == "sqrt_transform":
                c = r["col"]
                lines.append(f"df['{c}_sqrt'] = np.sqrt(df['{c}'].clip(lower=0))")
            elif t == "interaction":
                a, b, op = r["col_a"], r["col_b"], r.get("op","multiply")
                new_col  = f"{a}_{op[:3]}_{b}"
                ops_map  = {"multiply": f"df['{a}'] * df['{b}']",
                            "divide":   f"df['{a}'] / df['{b}'].replace(0, np.nan)",
                            "add":      f"df['{a}'] + df['{b}']",
                            "subtract": f"df['{a}'] - df['{b}']"}
                lines.append(f"df['{new_col}'] = {ops_map.get(op, '')}")
            elif t == "drop_duplicates":
                lines.append("df = df.drop_duplicates()")
            elif t == "drop_column":
                lines.append(f"df.drop(columns=['{r['col']}'], inplace=True)")
            elif t == "rename":
                lines.append(f"df.rename(columns={{'{r['col']}': '{r['new_name']}'}}, inplace=True)")
        lines.append("")

    if pipeline.get("encoding"):
        lines.append("# ── Encoding ──")
        for r in pipeline["encoding"]:
            col, m = r["col"], r["method"]
            if m == "label":
                lines.append(f"df['{col}'], _ = pd.factorize(df['{col}'])")
            elif m == "onehot":
                lines.append(f"df = pd.get_dummies(df, columns=['{col}'], prefix='{col}')")
            elif m == "drop":
                lines.append(f"df.drop(columns=['{col}'], inplace=True)")
        lines.append("")

    if pipeline.get("scaling"):
        lines.append("# ── Scaling ──")
        for r in pipeline["scaling"]:
            col, m = r["col"], r["method"]
            if m == "standard":
                lines.append(f"df['{col}'] = (df['{col}'] - df['{col}'].mean()) / df['{col}'].std()")
            elif m == "minmax":
                lines.append(f"df['{col}'] = (df['{col}'] - df['{col}'].min()) / (df['{col}'].max() - df['{col}'].min())")
            elif m == "robust":
                lines.append(f"df['{col}'] = (df['{col}'] - df['{col}'].median()) / (df['{col}'].quantile(0.75) - df['{col}'].quantile(0.25))")
        lines.append("")

    lines.append("# Save cleaned dataset")
    lines.append("df.to_csv('cleaned_dataset.csv', index=False)")
    lines.append("print(f'Done. Shape: {df.shape}')")

    return "\n".join(lines)
