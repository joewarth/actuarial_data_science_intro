"""
utils.py: Shared helper functions for Actuarial Data Science Presentation.

Usage:
from utils import *
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna

from pandas.api.types import is_numeric_dtype, CategoricalDtype
from sklearn.model_selection import StratifiedShuffleSplit, KFold

def fmt_cap(cap):
    """Format a numeric cap into a compact label (e.g., 100000 -> '100K', 1000000 -> '1MIL')."""
    if cap is None:
        return "uncapped"
    cap = int(cap)
    if cap >= 1_000_000:
        return f"{cap//1_000_000}MIL"
    if cap >= 1_000:
        return f"{cap//1_000}K"
    return str(cap)


def _normalize_cap_label(cap):
    """Normalize a cap input into the suffix used by capped columns (e.g., 100000 -> '100K', '1m' -> '1MIL')."""
    if cap is None:
        return None
    if isinstance(cap, (int, float)) and not np.isnan(cap):
        cap = int(cap)
        if cap >= 1_000_000:
            return f"{cap // 1_000_000}MIL"
        if cap >= 1_000:
            return f"{cap // 1_000}K"
        return str(cap)

    s = str(cap).upper().replace(" ", "").replace("MN", "M").replace("MM", "M")
    s = s.replace("MILLION", "MIL").replace("MIO", "MIL")
    # Treat bare 'M' as 'MIL'
    s = s.replace("MIL", "MIL").replace("M", "MIL").replace("K", "K")
    return s


def _resolve_claim_col(df, cap):
    """Return (claim_column_name, cap_label) given a desired cap, raising if no match exists."""
    if cap is None:
        return "ClaimAmount", "uncapped"

    label = _normalize_cap_label(cap)
    target = f"ClaimAmount_capped_{label}"
    if target in df.columns:
        return target, label

    candidates = [c for c in df.columns if c.startswith("ClaimAmount_capped_")]
    matches = [c for c in candidates if c.endswith(label)]
    if matches:
        return matches[0], label

    raise ValueError(
        f"Could not find capped column for cap='{cap}'. "
        f"Available: {', '.join(candidates[:8]) + (' ...' if len(candidates)>8 else '')}"
    )


def runmultiplot(data, dimension, metric="Frequency", cap=None, nstd_max=1, figsize=(20, 13)):
    """Plot exposure by a dimension with a line for Frequency/Severity/Pure Premium and optional SE bands.

    If `dimension` is numeric, it is auto-binned into quantile bins for readability.
    """
    metric = metric.strip().title()
    if metric not in {"Frequency", "Severity", "Pure Premium"}:
        raise ValueError("metric must be one of {'Frequency','Severity','Pure Premium'}")

    df = data.copy()

    # Resolve claim column based on cap (only needed for Severity / Pure Premium)
    claim_col, cap_label = ("ClaimAmount", "uncapped")
    if metric in {"Severity", "Pure Premium"}:
        claim_col, cap_label = _resolve_claim_col(df, cap)

    # Auto-bin numeric dimension (preserve category order)
    dim_col = dimension
    bin_label_order = None
    if is_numeric_dtype(df[dimension]):
        uniq = df[dimension].dropna().unique()
        if len(uniq) > 1:
            binned = pd.qcut(df[dimension], q=min(12, len(uniq)), duplicates="drop")
            if isinstance(binned.dtype, CategoricalDtype):
                cats = list(binned.cat.categories)
                labels = [str(iv) for iv in cats]
                bin_label_order = labels[:]
                binned = binned.cat.rename_categories(labels)
            df["_dim_binned"] = binned.astype(str).fillna("NA")
        else:
            df["_dim_binned"] = df[dimension].astype(str).fillna("NA")
            bin_label_order = pd.Index(df["_dim_binned"]).drop_duplicates().tolist()
        dim_col = "_dim_binned"

    # Per-row metric (for SE calc)
    if metric == "Frequency":
        df["_metric_row"] = np.divide(
            df["ClaimNb"], df["Exposure"],
            out=np.full(len(df), np.nan, dtype=float),
            where=(df["Exposure"].to_numpy(dtype=float) != 0)
        )
    elif metric == "Severity":
        df["_metric_row"] = np.divide(
            df[claim_col], df["ClaimNb"],
            out=np.full(len(df), np.nan, dtype=float),
            where=(df["ClaimNb"].to_numpy(dtype=float) != 0)
        )
    else:  # Pure Premium
        df["_metric_row"] = np.divide(
            df[claim_col], df["Exposure"],
            out=np.full(len(df), np.nan, dtype=float),
            where=(df["Exposure"].to_numpy(dtype=float) != 0)
        )

    # Aggregate
    agg = {
        "Exposure": "sum",
        "ClaimNb": "sum",
        claim_col: "sum",
        "_metric_row": ["mean", "std", "count"],
    }
    temp = (
        df.groupby(dim_col, dropna=False, observed=False)
          .agg(agg)
          .reset_index()
    )
    temp.columns = [c if isinstance(c, str) else "_".join([p for p in c if p]) for c in temp.columns]

    # Group-level metric + SE
    if metric == "Frequency":
        temp["Metric"] = np.divide(
            temp["ClaimNb_sum"], temp["Exposure_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["Exposure_sum"].to_numpy(dtype=float) != 0)
        )
        temp["SE"] = np.divide(
            np.sqrt(temp["ClaimNb_sum"]), temp["Exposure_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["Exposure_sum"].to_numpy(dtype=float) != 0)
        )
    elif metric == "Severity":
        temp["Metric"] = np.divide(
            temp[f"{claim_col}_sum"], temp["ClaimNb_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["ClaimNb_sum"].to_numpy(dtype=float) != 0)
        )
        temp["SE"] = np.divide(
            temp["_metric_row_std"], np.sqrt(temp["_metric_row_count"].clip(lower=1)),
            out=np.zeros(len(temp), dtype=float),
            where=(temp["_metric_row_count"].to_numpy(dtype=float) > 0)
        )
    else:  # Pure Premium
        temp["Metric"] = np.divide(
            temp[f"{claim_col}_sum"], temp["Exposure_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["Exposure_sum"].to_numpy(dtype=float) != 0)
        )
        temp["SE"] = np.divide(
            temp["_metric_row_std"], np.sqrt(temp["_metric_row_count"].clip(lower=1)),
            out=np.zeros(len(temp), dtype=float),
            where=(temp["_metric_row_count"].to_numpy(dtype=float) > 0)
        )

    # Portfolio line
    exposure_sum = df["Exposure"].sum()
    claimnb_sum = df["ClaimNb"].sum()
    claim_sum = df[claim_col].sum()
    if metric == "Frequency":
        portfolio_metric = (claimnb_sum / exposure_sum) if exposure_sum else 0.0
    elif metric == "Severity":
        portfolio_metric = (claim_sum / claimnb_sum) if claimnb_sum else 0.0
    else:
        portfolio_metric = (claim_sum / exposure_sum) if exposure_sum else 0.0

    # X order & ranks
    if dim_col == "_dim_binned" and bin_label_order is not None:
        order = bin_label_order
    else:
        order = pd.Index(temp[dim_col].astype(str)).drop_duplicates().tolist()

    temp = temp.set_index(dim_col).loc[order].reset_index()
    temp["Rank"] = np.arange(len(temp))

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)

    sns.barplot(x=dim_col, y="Exposure_sum", data=temp, estimator=sum, order=order, alpha=0.7, ax=ax1)
    if ax1.containers:
        ax1.bar_label(ax1.containers[0])
    ax1.set_ylabel("Exposure")
    ax1.set_xlabel(dimension)
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")

    ax2 = ax1.twinx()
    ax2.set_zorder(ax1.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(metric)

    sns.lineplot(x="Rank", y="Metric", data=temp, marker="o", markersize=10, ax=ax2, label=metric)

    for n in range(1, nstd_max + 1):
        ax2.fill_between(
            temp["Rank"].to_numpy(),
            np.maximum((temp["Metric"] - n * temp["SE"]).to_numpy(), 0.0),
            np.maximum((temp["Metric"] + n * temp["SE"]).to_numpy(), 0.0),
            alpha=0.25,
            label=(f"±{n}·SE"),
        )

    ax2.axhline(y=portfolio_metric, linestyle="--", linewidth=2, label="Portfolio")

    # Align tick positions with labels (Rank-based line over bar centers)
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_xticks(temp["Rank"])
    ax1.set_xticklabels(order, rotation=45, ha="right")

    # Legend (dedup)
    handles, labels = ax2.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    ax2.legend(h2, l2, loc="upper left")

    title_cap = "" if (metric == "Frequency" or cap_label == "uncapped") else f" (cap {cap_label})"
    plt.title(f"{metric}{title_cap} by {dimension if dim_col==dimension else f'{dimension} (binned)'} vs Portfolio")
    plt.tight_layout()

    return fig


def stratified_split_match_portfolio_freq(
    df,
    group_col="IDpol",
    exposure_col="Exposure",
    claim_col="ClaimNb",
    test_size=0.20,
    q=10,
    tol=0.005,
    max_tries=200,
    random_state=42,
):
    """Create a train/test split stratified by policy-level frequency and matched on portfolio frequency."""
    req = {group_col, exposure_col, claim_col}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Collapse to one row per policy (unit for stratification)
    g = (
        df.groupby(group_col, as_index=True)
          .agg(Exposure_sum=(exposure_col, "sum"), ClaimNb_sum=(claim_col, "sum"))
    )
    g = g[g["Exposure_sum"] > 0].copy()
    if g.empty:
        raise ValueError("No policies with positive exposure after aggregation.")

    g["pol_freq"] = g["ClaimNb_sum"] / g["Exposure_sum"]

    def make_bins(g_, q_try):
        return pd.qcut(g_["pol_freq"], q=min(q_try, g_["pol_freq"].nunique()), duplicates="drop")

    def bins_ok(bins, test_size_):
        vc = bins.value_counts()
        return (len(vc) >= 2) and all((vc * test_size_ >= 1) & (vc * (1 - test_size_) >= 1))

    q_try = min(q, g["pol_freq"].nunique())
    bins = make_bins(g, q_try)
    while not bins_ok(bins, test_size):
        q_try -= 1
        if q_try < 2:
            bins = (g["ClaimNb_sum"] > 0).astype(int)  # fallback
            break
        bins = make_bins(g, q_try)

    labels = bins.astype(str)
    target_pf = g["ClaimNb_sum"].sum() / g["Exposure_sum"].sum()

    def portfolio_freq(sub):
        return sub["ClaimNb_sum"].sum() / sub["Exposure_sum"].sum()

    best = None
    best_diff = float("inf")
    seed = random_state

    for _ in range(max_tries):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(g)), labels))

        tr, te = g.iloc[train_idx], g.iloc[test_idx]
        tr_pf, te_pf = portfolio_freq(tr), portfolio_freq(te)
        diff = abs(tr_pf - te_pf)
        if diff < best_diff:
            best_diff = diff
            best = (tr.index.values, te.index.values, tr_pf, te_pf, seed, q_try)
            if diff <= tol:
                break
        seed += 1

    if best is None:
        raise RuntimeError("Failed to produce a split. Try increasing max_tries or relaxing tol.")

    train_pols, test_pols, tr_pf, te_pf, used_seed, used_q = best

    out = df.copy()
    out["set"] = np.where(out[group_col].isin(test_pols), "test", "train")

    print(f"Used seed: {used_seed} | bins: {used_q if used_q>=2 else 'has_claim'}")
    print(f"Overall PF: {target_pf:.6f}")
    print(f"Train PF  : {tr_pf:.6f}")
    print(f"Test  PF  : {te_pf:.6f}")
    print(f"|Train-Test|: {abs(tr_pf - te_pf):.6f} (tol={tol})")

    return out


def make_X_y(df, cat_cols, num_cols, y_col, exposure_col="Exposure"):
    """Build (X, y, w, base_score) for XGBoost Tweedie with native categorical support."""
    X = df[cat_cols + num_cols].copy()

    # Categoricals: category dtype + explicit missing bucket
    for c in cat_cols:
        X[c] = X[c].astype("category")
        if X[c].isna().any():
            X[c] = X[c].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

    # Numerics: coerce -> float, then clean numeric cols
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # y and weights
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(df[exposure_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Exposure-weighted base_score in log-space
    base_score_untrans = (w * y).sum() / max(w.sum(), 1e-12)
    base_score = float(np.log(base_score_untrans + 1e-12))

    return X, y, w, base_score


def tune_xgb_tweedie_optuna(
    df,
    cat_cols,
    num_cols,
    y_col,
    exposure_col="Exposure",
    n_splits=5,
    n_trials=50,
    random_state=42,
    num_boost_round_max=5000,
    early_stopping_rounds=50,
):
    """Tune an XGBoost Tweedie model with Optuna using K-fold CV on tweedie nloglik."""
    # Preprocess (native categorical)
    X = df[cat_cols + num_cols].copy()

    for c in cat_cols:
        X[c] = X[c].astype("category")
        if X[c].isna().any():
            X[c] = X[c].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(df[exposure_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    base_score_untrans = (w * y).sum() / max(w.sum(), 1e-12)
    base_score = float(np.log(base_score_untrans + 1e-12))

    train_categories = {c: X[c].cat.categories for c in cat_cols}

    dtrain = xgb.DMatrix(X, label=y, weight=w, enable_categorical=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [(tr_idx, va_idx) for tr_idx, va_idx in kf.split(X)]

    def _get_test_metric_col(cv_df):
        candidates = [c for c in cv_df.columns if c.startswith("test-") and c.endswith("-mean")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly 1 test metric mean column, found: {candidates}")
        return candidates[0]

    def objective(trial):
        p = trial.suggest_float("tweedie_variance_power", 1.2, 1.9, step=0.05)
        eval_metric = f"tweedie-nloglik@{p:.2f}"

        params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": p,
            "eval_metric": eval_metric,
            "base_score": base_score,
            "tree_method": "hist",
            "seed": random_state,

            "eta": trial.suggest_float("eta", 0.01, 0.15, step=0.01),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100000),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0, step=0.5),

            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),

            "lambda": trial.suggest_float("lambda", 0.0, 50.0, step=1.0),
            "alpha": trial.suggest_float("alpha", 0.0, 10.0, step=0.5),

            "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0, step=0.5),

            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 16),
            "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 256),
        }

        cv = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round_max,
            folds=folds,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        test_col = _get_test_metric_col(cv)
        return float(cv[test_col].min())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    p_best = best_trial.params["tweedie_variance_power"]
    eval_metric_best = f"tweedie-nloglik@{p_best}"

    best_params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": p_best,
        "eval_metric": eval_metric_best,

        "base_score": base_score,
        "tree_method": "hist",
        "seed": random_state,

        "eta": best_trial.params["eta"],
        "max_depth": best_trial.params["max_depth"],
        "min_child_weight": best_trial.params["min_child_weight"],
        "gamma": best_trial.params["gamma"],

        "subsample": best_trial.params["subsample"],
        "colsample_bytree": best_trial.params["colsample_bytree"],

        "lambda": best_trial.params["lambda"],
        "alpha": best_trial.params["alpha"],

        "max_delta_step": best_trial.params["max_delta_step"],
        "max_cat_to_onehot": best_trial.params["max_cat_to_onehot"],
        "max_cat_threshold": best_trial.params["max_cat_threshold"],
    }

    cv_best = xgb.cv(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round_max,
        folds=folds,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_num_boost_round = int(cv_best.shape[0])

    return study, best_params, best_num_boost_round, train_categories


def preprocess_for_predict(df_new, cat_cols, num_cols, train_categories=None):
    """Preprocess a new DataFrame for XGBoost native categorical prediction."""
    X_new = df_new[cat_cols + num_cols].copy()

    # Categoricals
    for c in cat_cols:
        X_new[c] = X_new[c].astype("category")

        if train_categories is not None:
            X_new[c] = X_new[c].cat.set_categories(train_categories[c])
            if "__MISSING__" in train_categories[c]:
                X_new[c] = X_new[c].fillna("__MISSING__")
        else:
            if X_new[c].isna().any():
                X_new[c] = X_new[c].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

    # Numerics
    for c in num_cols:
        X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
    X_new[num_cols] = X_new[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_new


def assign_weighted_deciles(score, weight, n=10):
    """Assign 1..n deciles so each bucket has ~equal total weight (exposure)."""
    s = pd.DataFrame({"score": score, "w": weight}).sort_values("score").reset_index()
    s["cw"] = s["w"].cumsum()
    total = s["w"].sum()
    s["decile"] = np.ceil(n * s["cw"] / total).astype(int).clip(1, n)
    out = pd.Series(index=s["index"], data=s["decile"].values)
    return out.reindex(score.index)


def lift_table_by_decile(df, y_pp_col, pred_pp, exposure_col="Exposure", n_deciles=10):
    """Create an exposure-weighted lift table by predicted pure premium deciles."""
    d = df.copy()
    d = d.loc[d[exposure_col] > 0].copy()

    d["pred_pp"] = pred_pp
    d["actual_loss"] = d[y_pp_col] * d[exposure_col]
    d["pred_loss"] = d["pred_pp"] * d[exposure_col]

    d["decile"] = assign_weighted_deciles(d["pred_pp"], d[exposure_col], n=n_deciles)

    g = (
        d.groupby("decile", as_index=False)
         .agg(
             exposure=(exposure_col, "sum"),
             actual_loss=("actual_loss", "sum"),
             pred_loss=("pred_loss", "sum"),
         )
    )

    g["actual_pp"] = g["actual_loss"] / g["exposure"]
    g["pred_pp"] = g["pred_loss"] / g["exposure"]

    port_actual_pp = d["actual_loss"].sum() / d[exposure_col].sum()
    port_pred_pp = d["pred_loss"].sum() / d[exposure_col].sum()

    g["actual_lift"] = g["actual_pp"] / port_actual_pp
    g["pred_lift"] = g["pred_pp"] / port_pred_pp

    return g, port_actual_pp, port_pred_pp


def plot_lift(g, title="Lift Chart (Exposure-Weighted Deciles)"):
    """Plot actual vs predicted lift lines from a lift table."""
    plt.figure()
    plt.plot(g["decile"], g["actual_lift"], marker="o", label="Actual lift")
    plt.plot(g["decile"], g["pred_lift"], marker="o", label="Predicted lift")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Prediction decile (1 = lowest predicted risk, 10 = highest)")
    plt.ylabel("Relativity vs portfolio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def orderedLorenz(actual_losses, current_premium, new_premium, plot=True, sample_prop=0.1):
    """Compute an ordered Lorenz/Gini-style index comparing new premium vs current premium vs losses.

    Sorts risks by the premium ratio (new/current), then computes a Gini-style area measure.
    Optionally plots a sampled Lorenz curve.
    """
    dt = pd.DataFrame({"y": actual_losses, "b": current_premium, "p": new_premium}).dropna(subset=["y", "b", "p"])

    dt["r"] = dt["p"] / dt["b"]
    dt = dt.sort_values("r", kind="mergesort").reset_index(drop=True)

    dt["p_dist"] = dt["b"].cumsum() / dt["b"].sum()
    dt["l_dist"] = dt["y"].cumsum() / dt["y"].sum()

    zero = pd.DataFrame([{col: 0 for col in dt.columns}])
    dt0 = pd.concat([zero, dt], ignore_index=True)

    l = dt0["l_dist"].to_numpy()
    p = dt0["p_dist"].to_numpy()

    l_shift = np.roll(l, 1)
    p_shift = np.roll(p, 1)
    l_shift[0] = np.nan
    p_shift[0] = np.nan

    area = np.nansum((l + l_shift) * (p - p_shift))
    gini = 1.0 - area

    if plot:
        n = len(dt0)
        m = int(sample_prop * n)
        sample_idx = np.arange(n) if m <= 0 else np.random.choice(np.arange(n), size=m, replace=False)
        sample_dt = dt0.iloc[np.sort(sample_idx)]

        plt.figure()
        plt.plot(sample_dt["p_dist"], sample_dt["l_dist"])
        plt.plot([0, 1], [0, 1])
        plt.title(f"gini index = {gini:0.2f}")
        plt.ylabel("loss distribution")
        plt.xlabel("premium distribution")
        plt.show()

    return gini