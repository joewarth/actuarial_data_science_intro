"""
utils.py: Shared helper functions for Insurance Pricing Project.

Summary:
Utilities that are dependency-heavy and very project-specific.
I make no claim that these are "pythontic" and well-written.

Usage:
from utils import *
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
import patsy

from pandas.api.types import is_numeric_dtype, CategoricalDtype
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from patsy import dmatrix, build_design_matrices

def fmt_cap(cap):
    if cap is None: return "uncapped"
    cap = int(cap)
    if cap >= 1_000_000: return f"{cap//1_000_000}MIL"
    if cap >= 1_000:     return f"{cap//1_000}K"
    return str(cap)



def _normalize_cap_label(cap):
    """Return label suffix used in capped columns, e.g. 100000 -> '100K', '1M' -> '1MIL'."""
    if cap is None:
        return None
    if isinstance(cap, (int, float)) and not np.isnan(cap):
        cap = int(cap)
        if cap >= 1_000_000:
            return f"{cap // 1_000_000}MIL"
        elif cap >= 1_000:
            return f"{cap // 1_000}K"
        else:
            return str(cap)
    s = str(cap).upper().replace(" ", "").replace("MN", "M").replace("MM", "M")
    s = s.replace("MILLION", "MIL").replace("MIO", "MIL")
    s = s.replace("MIL", "MIL").replace("M", "MIL").replace("K", "K")
    return s

def _resolve_claim_col(df, cap):
    """Pick ClaimAmount or a capped column based on 'cap'."""
    if cap is None:
        return "ClaimAmount", "uncapped"
    label = _normalize_cap_label(cap)
    target = f"ClaimAmount_capped_{label}"
    if target in df.columns:
        return target, label
    # fallback: try to find a best-effort match
    candidates = [c for c in df.columns if c.startswith("ClaimAmount_capped_")]
    # try suffix match (e.g., endswith '100K' or '1MIL')
    matches = [c for c in candidates if c.endswith(label)]
    if matches:
        return matches[0], label
    raise ValueError(
        f"Could not find capped column for cap='{cap}'. "
        f"Available: {', '.join(candidates[:8]) + (' ...' if len(candidates)>8 else '')}"
    )

def runmultiplot(
    data: pd.DataFrame,
    dimension: str,
    metric: str = "Frequency",
    cap=None,
    nstd_max: int = 1,
    figsize=(20, 13)
):
    """
    Plot Exposure bars by `dimension` with a line for the chosen metric and n-std bands.
    If `dimension` is numeric, it will be auto-binned (quantile bins).
    """
    metric = metric.strip().title()
    if metric not in {"Frequency", "Severity", "Pure Premium"}:
        raise ValueError("metric must be one of {'Frequency','Severity','Pure Premium'}")

    df = data.copy()

    # --- resolve claim column based on cap
    claim_col, cap_label = ("ClaimAmount", "uncapped")
    if metric in {"Severity", "Pure Premium"}:
        _resolver = globals().get("_resolve_claim_col", None)
        if cap is not None and callable(_resolver):
            claim_col, cap_label = _resolver(df, cap)
        elif cap is not None:
            cap_str = str(cap).upper().replace(",", "").replace("_", "").replace(" ", "")
            mapping = {"100K": "100000", "1M": "1000000", "1MIL": "1000000"}
            cap_norm = mapping.get(cap_str, cap_str)
            cand = f"ClaimAmount_capped_{cap_norm}"
            claim_col, cap_label = (cand, cap_norm) if cand in df.columns else ("ClaimAmount", "uncapped")

    # --- AUTO-BIN if numeric (preserve category order)
    dim_col = dimension
    bin_label_order = None  # will hold the ordered labels for plotting
    if is_numeric_dtype(df[dimension]):
        uniq = df[dimension].dropna().unique()
        if len(uniq) > 1:
            binned = pd.qcut(df[dimension], q=min(12, len(uniq)), duplicates="drop")
            if isinstance(binned.dtype, CategoricalDtype):
                cats = list(binned.cat.categories)          # ordered IntervalIndex
                labels = [str(iv) for iv in cats]           # unique, readable
                bin_label_order = labels[:]                  # preserve order for plotting
                binned = binned.cat.rename_categories(labels)
            df["_dim_binned"] = binned.astype(str).fillna("NA")
        else:
            df["_dim_binned"] = df[dimension].astype(str).fillna("NA")
            bin_label_order = pd.Index(df["_dim_binned"]).drop_duplicates().tolist()
        dim_col = "_dim_binned"

    # --- per-row metric (for SE calc)
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

    # --- aggregate
    agg = {
        "Exposure": "sum",
        "ClaimNb": "sum",
        claim_col: "sum",
        "_metric_row": ["mean", "std", "count"]
    }
    temp = (
        df.groupby(dim_col, dropna=False, observed=False)
          .agg(agg)
          .reset_index()
    )
    temp.columns = [c if isinstance(c, str) else "_".join([p for p in c if p]) for c in temp.columns]

    # --- group-level metric + SE
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

    # --- portfolio line
    exposure_sum = df["Exposure"].sum()
    claimnb_sum  = df["ClaimNb"].sum()
    claim_sum    = df[claim_col].sum()
    if metric == "Frequency":
        portfolio_metric = (claimnb_sum / exposure_sum) if exposure_sum else 0.0
    elif metric == "Severity":
        portfolio_metric = (claim_sum / claimnb_sum) if claimnb_sum else 0.0
    else:
        portfolio_metric = (claim_sum / exposure_sum) if exposure_sum else 0.0

    # --- x order & ranks
    if dim_col == "_dim_binned" and bin_label_order is not None:
        order = bin_label_order  # interval order left->right
    else:
        order = pd.Index(temp[dim_col].astype(str)).drop_duplicates().tolist()

    temp = temp.set_index(dim_col).loc[order].reset_index()
    temp["Rank"] = np.arange(len(temp))

    # --- plot
    fig, ax1 = plt.subplots(figsize=figsize)

    sns.barplot(x=dim_col, y="Exposure_sum", data=temp, estimator=sum, order=order, alpha=0.7, ax=ax1)
    if ax1.containers:
        ax1.bar_label(ax1.containers[0])
    ax1.set_ylabel("Exposure")
    ax1.set_xlabel(dimension)  # <- keep axis label as the original column name
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
            label=(f"±{n}·SE")
        )

    ax2.axhline(y=portfolio_metric, linestyle="--", linewidth=2, label="Portfolio")

    # align tick positions with labels (Rank-based line over bar centers)
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_xticks(temp["Rank"])
    ax1.set_xticklabels(order, rotation=45, ha="right")

    # legend (dedup)
    handles, labels = ax2.get_legend_handles_labels()
    seen = set(); h2=[]; l2=[]
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax2.legend(h2, l2, loc="upper left")

    title_cap = "" if (metric == "Frequency" or cap_label == "uncapped") else f" (cap {cap_label})"
    plt.title(f"{metric}{title_cap} by {dimension if dim_col==dimension else f'{dimension} (binned)'} vs Portfolio")
    plt.tight_layout()

    return fig

def stratified_split_match_portfolio_freq(
    df: pd.DataFrame,
    group_col: str = "IDpol",
    exposure_col: str = "Exposure",
    claim_col: str = "ClaimNb",
    test_size: float = 0.20,
    q: int = 10,                 # initial number of bins
    tol: float = 0.005,          # |train_pf - test_pf| tolerance
    max_tries: int = 200,
    random_state: int = 42
):
    req = {group_col, exposure_col, claim_col}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Collapse to one row per policy just in case
    g = (df.groupby(group_col, as_index=True)
           .agg(Exposure_sum=(exposure_col, "sum"),
                ClaimNb_sum=(claim_col, "sum")))
    g = g[g["Exposure_sum"] > 0].copy()
    if g.empty:
        raise ValueError("No policies with positive exposure after aggregation.")

    g["pol_freq"] = g["ClaimNb_sum"] / g["Exposure_sum"]

    # Find a valid number of bins (>=2) so each bin can allocate to train/test
    def make_bins(g, q_try):
        bins = pd.qcut(g["pol_freq"], q=min(q_try, g["pol_freq"].nunique()),
                       duplicates="drop")
        return bins

    def bins_ok(bins, test_size):
        # need at least one item in train and test for every class
        vc = bins.value_counts()
        return (len(vc) >= 2) and all((vc * test_size >= 1) & (vc * (1 - test_size) >= 1))

    q_try = min(q, g["pol_freq"].nunique())
    bins = make_bins(g, q_try)
    while not bins_ok(bins, test_size):
        q_try -= 1
        if q_try < 2:
            # fall back to a single strat variable (has_claim) if freq bins too sparse
            bins = (g["ClaimNb_sum"] > 0).astype(int)
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
        # X is unused by SSS; pass zeros of the right length
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

def make_pf_balanced_group_folds(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    bins: int = 10,          # number of quantile bins for policy-level frequency
    tol: float = 0.002,      # acceptable max |pf_i - pf_j| across folds
    max_tries: int = 200,
    random_state: int = 42,
):
    """
    Returns:
      fold_id_train : np.ndarray of length len(train_df) with values {0..n_splits-1}
      cv_iterable   : list of (train_idx, val_idx) index arrays aligned to train_df rows
      fold_summary  : DataFrame with exposure, claims, portfolio freq by fold
    """
    req = {"IDpol", "Exposure", "ClaimNb"}
    missing = req - set(train_df.columns)
    if missing:
        raise KeyError(f"Missing columns in train_df: {missing}")

    # --- Aggregate to one row per policy (the unit we stratify on)
    g = (train_df.groupby("IDpol", as_index=True)
                   .agg(Exposure_sum=("Exposure","sum"),
                        ClaimNb_sum=("ClaimNb","sum")))
    g = g[g["Exposure_sum"] > 0].copy()
    g["pol_freq"] = g["ClaimNb_sum"] / g["Exposure_sum"]

    # Bin policy-level frequency for stratification (helps balance folds)
    nbins = min(bins, g["pol_freq"].nunique() or 1)
    if nbins < 2:
        # degenerate case: almost identical freqs; just treat as one bin
        g["freq_bin"] = "all"
    else:
        g["freq_bin"] = pd.qcut(g["pol_freq"], q=nbins, duplicates="drop").astype(str)

    pol_ids = g.index.to_numpy()
    y_bins  = g["freq_bin"].to_numpy()

    # Helper to compute fold portfolio freq quickly
    def fold_pf(assign):
        # assign: array mapping policies -> fold id
        summaries = []
        for f in range(n_splits):
            pol_in_fold = pol_ids[assign == f]
            sub = g.loc[pol_in_fold]
            expo = sub["Exposure_sum"].sum()
            clm  = sub["ClaimNb_sum"].sum()
            pf = (clm / expo) if expo > 0 else 0.0
            summaries.append((expo, clm, pf))
        pfs = [pf for _,_,pf in summaries]
        return max(pfs) - min(pfs), summaries

    # Try different seeds: stratified K-fold over policies by frequency bins
    best = None
    best_spread = float("inf")
    seed = random_state
    for _ in range(max_tries):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # skf.split expects arrays aligned to policies
        # X is unused -> pass zeros
        assign = np.empty(len(pol_ids), dtype=int)
        # Build folds once; StratifiedKFold yields n_splits splits
        for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(pol_ids)), y_bins)):
            assign[val_idx] = fold_id

        spread, summaries = fold_pf(assign)
        if spread < best_spread:
            best_spread = spread
            best = (assign.copy(), summaries, seed)
            if spread <= tol:
                break
        seed += 1

    if best is None:
        raise RuntimeError("Failed to create folds; try increasing max_tries or relaxing tol.")

    assign, summaries, used_seed = best

    # Map policies -> fold id for each row in train_df
    pol_to_fold = dict(zip(pol_ids, assign))
    fold_id_train = train_df["IDpol"].map(pol_to_fold).to_numpy()

    # Build cv_iterable aligned to train_df row order
    idx = np.arange(len(train_df))
    cv_iterable = []
    for f in range(n_splits):
        val_idx = idx[fold_id_train == f]
        trn_idx = idx[fold_id_train != f]
        if len(val_idx) == 0 or len(trn_idx) == 0:
            continue
        cv_iterable.append((trn_idx, val_idx))

    # Nice summary table
    fold_summary = pd.DataFrame(
        [{"fold": f, "exposure": ex, "claims": cl, "portfolio_freq": pf}
         for f, (ex, cl, pf) in enumerate(summaries)]
    )
    print(f"CV folds seed: {used_seed} | PF spread across folds: {best_spread:.6f} (tol={tol})")
    return fold_id_train, cv_iterable, fold_summary

def tweedie_profile_mle(
    X, y, exposure=None, weights=None,
    p_bounds=(1.01, 1.99),  # common for compound Poisson-gamma; adjust if needed
    p_grid=np.linspace(1.1, 1.9, 17),  # coarse grid to locate a good bracket
    maxiter=100
):
    """
    Returns: dict with p_hat, result (GLMResults at p_hat), llf_path (grid),
             p_refined (scalar-refined), result_refined (optional refined fit)
    """
    X_ = sm.add_constant(X, has_constant='add')
    offset = np.log(exposure) if exposure is not None else None

    def fit_and_llf(p):
        fam = sm.families.Tweedie(var_power=p, link=sm.families.links.Log())
        mod = sm.GLM(y, X_, family=fam, offset=offset, freq_weights=weights)
        res = mod.fit(maxiter=maxiter, disp=0)
        return res, res.llf

    # 1) coarse grid search
    llfs = []
    for p in p_grid:
        try:
            _, ll = fit_and_llf(p)
        except Exception:
            ll = -np.inf
        llfs.append(ll)

    p0 = p_grid[np.argmax(llfs)]

    # 2) refine with a 1D scalar optimizer around the best grid point
    #    we *maximize* llf, so minimize negative llf
    def neg_llf(p):
        # keep p within Tweedie’s valid range
        if p <= 0: return np.inf
        try:
            _, ll = fit_and_llf(p)
            return -ll
        except Exception:
            return np.inf

    bracket_lo = max(p_bounds[0], p0 - 0.2)
    bracket_hi = min(p_bounds[1], p0 + 0.2)
    opt = minimize_scalar(neg_llf, bounds=(bracket_lo, bracket_hi), method='bounded')
    p_refined = float(opt.x)

    # 3) final fit at refined p
    res_refined, _ = fit_and_llf(p_refined)

    return {
        "p_hat": p0,
        "result": None,  # optional: fit again at p0 if you want both
        "llf_path": (p_grid, np.array(llfs)),
        "p_refined": p_refined,
        "result_refined": res_refined
    }

def phat_for_combo(thr, cap, df, num_cols):
    d = df.loc[df["Exposure"] >= thr]
    X = d[num_cols].to_numpy(dtype=float)
    y = d[f"pure_premium_capped_{fmt_cap(cap)}"].to_numpy(dtype=float)
    w = d["Exposure"].to_numpy(dtype=float)

    out = tweedie_profile_mle(
        X=X, y=y, weights=w,
        p_bounds=(1.05, 1.95),
        p_grid=np.linspace(1.2, 1.8, 9),  # narrower grid
        maxiter=100                       # fewer IRLS steps
    )
    return {"cap": cap, "min_exposure_threshold": float(thr),
            "p_hat": round(out["p_refined"], 2)}

def weighteddev_tweedie(data=None, obs=None, pred=None, testpower=None, weights=None):
    """
    Compute 2 * sum(weights * dev_i) where
      dev_i = y * ((y^(1-p) - mu^(1-p)) / (1-p)) - (y^(2-p) - mu^(2-p)) / (2-p)
    with the convention y^a = 0 when y == 0 (same behavior as your R ifelse).

    Parameters
    ----------
    data : pd.DataFrame or None
        Optional DataFrame source if passing column names.
    obs, pred, weights : str or array-like
        If `data` is given, these can be column names (strings). Otherwise pass arrays.
    testpower : float
        Tweedie power p (must not be 1 or 2).

    Returns
    -------
    float
        Total weighted deviance (same scale as your R function).
    """
    if testpower is None:
        raise ValueError("testpower (p) must be provided.")
    p = float(testpower)

    # Guard against p=1 or p=2 (division by zero in the deviance formula)
    eps = 1e-12
    if abs(p - 1.0) < eps or abs(p - 2.0) < eps:
        raise ValueError("testpower must not be 1 or 2.")

    def get_series(x):
        if isinstance(x, str):
            if data is None:
                raise ValueError(f"Got column name '{x}' but no DataFrame `data` was provided.")
            return pd.Series(data[x])
        return pd.Series(x)

    y = get_series(obs).astype(float)
    mu = get_series(pred).astype(float)
    w  = get_series(weights).astype(float)

    # Match your R ifelse: when value is 0, use 0 instead of value**power
    y_1mp = np.where(y == 0, 0.0, np.power(y, 1.0 - p))
    mu_1mp = np.where(mu == 0, 0.0, np.power(mu, 1.0 - p))
    y_2mp = np.where(y == 0, 0.0, np.power(y, 2.0 - p))
    mu_2mp = np.where(mu == 0, 0.0, np.power(mu, 2.0 - p))

    dev_i = y * ((y_1mp - mu_1mp) / (1.0 - p)) - ((y_2mp - mu_2mp) / (2.0 - p))
    weighted_dev = w * dev_i

    return float(2.0 * np.sum(weighted_dev))

def evaluate_scenario(cap, min_exposure_threshold, p_hat, df, num_cols, uncapped_p_hat):
    devs = []
    for fold in sorted(df["fold_id"].unique()):
        # split
        fold_train = df[df["fold_id"] != fold].copy()
        fold_test  = df[df["fold_id"] == fold].copy()

        # filter train
        fold_train = fold_train[fold_train["Exposure"] >= min_exposure_threshold].copy()
        if fold_train.empty:
            continue  # nothing to fit on for this fold after filtering

        # model inputs
        X_train_raw = fold_train[num_cols].to_numpy(dtype=float)
        X_test_raw  = fold_test[num_cols].to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        y_train = fold_train[f"pure_premium_capped_{fmt_cap(cap)}"].to_numpy(dtype=float)
        w_train = fold_train["Exposure"].to_numpy(dtype=float)

        # fit
        fit = TweedieRegressor(power=p_hat, link="log", alpha=1e-6, max_iter=1000)
        fit.fit(X_train, y_train, sample_weight=w_train)

        # predict on test
        preds = fit.predict(X_test)

        # rebase to uncapped test average
        predicted_tot = (preds * fold_test["Exposure"]).sum() / (fold_test["Exposure"]).sum()
        uncapped_tot  = (fold_test["pure_premium_uncapped"] * fold_test["Exposure"]).sum() / (fold_test["Exposure"]).sum()
        avg_rebase = uncapped_tot / predicted_tot
        fold_test["fitted"] = preds * avg_rebase

        # deviance on holdout vs uncapped obs
        dev_fold = weighteddev_tweedie(
            data=fold_test,
            obs="pure_premium_uncapped",
            pred="fitted",
            testpower=uncapped_p_hat,   # e.g., 1.95
            weights="Exposure"
        )
        devs.append(dev_fold)

    return float(np.mean(devs)) if devs else np.nan

def weighted_percentiles(df, value_col, weight_col, probs):
    d = df[[value_col, weight_col]].dropna().copy()
    d[value_col] = d[value_col].astype(float)
    d[weight_col] = d[weight_col].astype(float)
    d = d[d[weight_col] > 0]
    if d.empty or d[weight_col].sum() == 0:
        return pd.Series([np.nan]*len(probs), index=probs)

    d = d.sort_values(value_col)
    x = d[value_col].to_numpy()
    w = d[weight_col].to_numpy()
    cdf = np.cumsum(w) / w.sum()

    # Linear interpolation on the weighted CDF
    return pd.Series(np.interp(probs, cdf, x), index=probs)

def make_knot_grid(candidates, knot_counts=(1, 2, 3), min_space=2, spacing_units="absolute"):
    """
    candidates : sorted 1-D array of candidate knot positions
    knot_counts : iterable of how many interior knots to choose (e.g., (1,2,3))
    min_space : minimum spacing between adjacent knots (default 2). If None, no spacing rule.
    spacing_units : 'absolute' or 'relative' (relative = fraction of data range)
    """
    cands = np.asarray(candidates, dtype=float)
    if cands.size == 0:
        return pd.DataFrame(columns=["n_knots", "knots"])

    # allow disabling spacing rule
    if min_space is not None and spacing_units == "relative":
        data_range = float(cands.max() - cands.min()) if cands.size > 1 else 0.0
        min_space = min_space * data_range

    rows = []
    for k in knot_counts:
        if k <= 0 or k > len(cands):
            continue
        for combo in itertools.combinations(cands, k):
            if min_space is not None and k > 1:
                if np.any(np.diff(combo) < min_space):
                    continue
            rows.append({"n_knots": k, "knots": combo})
    return pd.DataFrame(rows)


def expand_knots_df(df):
    max_k = df["n_knots"].max() if not df.empty else 0
    out = df.copy()
    for i in range(1, max_k + 1):
        out[f"knot{i}"] = out["knots"].apply(lambda ks: ks[i - 1] if len(ks) >= i else np.nan)
    cols = ["n_knots"] + [f"knot{i}" for i in range(1, max_k + 1)]
    return out[cols]

def row_knots(row, knot_cols=("knot1", "knot2", "knot3", "knot4"), n_col="n_knots"):
    n = int(row[n_col])
    ks = [row[c] for c in knot_cols[:n]]
    ks = [float(k) for k in ks if pd.notna(k)]
    return tuple(ks)

def design_formula(kind: str, x_col: str, knots=(), degree: int = 3) -> str:
    """
    Returns a Patsy formula string for the given scenario kind.
    - linear:      1 + x
    - quadratic:   1 + x + x^2
    - cubic:       1 + x + x^2 + x^3
    - spline:      1 + bs(x, knots=..., degree=...)
    """
    if kind == "linear":
        return f"1 + {x_col}"
    if kind == "quadratic":
        return f"1 + {x_col} + I({x_col}**2)"
    if kind == "cubic":
        return f"1 + {x_col} + I({x_col}**2) + I({x_col}**3)"
    if kind == "spline":
        return f"1 + bs({x_col}, knots={knots}, degree={degree}, include_intercept=False)"

    raise ValueError(f"Unknown kind: {kind}")

def evaluate_spline_scenario_cv(
    scenario: dict,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    p_fixed: float,
    weight_col: str = "Exposure",
    spline_degree: int = 3,
    alpha: float = 1e-6,
    max_iter: int = 1000,
):
    kind = scenario["kind"]
    knots = scenario.get("knots", ())

    # Guard: spline scenario but no knots
    if kind == "spline" and (knots is None or len(knots) == 0):
        return {**scenario, "score": np.nan}

    devs = []
    for fold in sorted(df["fold_id"].unique()):
        fold_train = df[df["fold_id"] != fold].copy()
        fold_test  = df[df["fold_id"] == fold].copy()
        if fold_train.empty or fold_test.empty:
            continue

        formula = design_formula(kind, x_col, knots=knots, degree=spline_degree)

        X_train = dmatrix(formula, fold_train, return_type="dataframe")
        design_info = X_train.design_info
        X_test = build_design_matrices([design_info], fold_test, return_type="dataframe")[0]
        y_train = fold_train[y_col].to_numpy(dtype=float)
        w_train = fold_train[weight_col].to_numpy(dtype=float)

        fit = TweedieRegressor(power=p_fixed, link="log", alpha=alpha, max_iter=max_iter)
        fit.fit(X_train.to_numpy(), y_train, sample_weight=w_train)

        mu_test = fit.predict(X_test.to_numpy())

        tmp = fold_test[[y_col, weight_col]].copy()
        tmp["pred"] = mu_test

        dev_fold = weighteddev_tweedie(
            data=tmp,
            obs=y_col,
            pred="pred",
            testpower=p_fixed,
            weights=weight_col,
        )
        devs.append(dev_fold)

    score = float(np.mean(devs)) if devs else np.nan
    return {**scenario, "score": score}

class PatsySplineTransformer(BaseEstimator, TransformerMixin):
    """
    Build a spline design matrix using a patsy formula (e.g., bs(...)).
    Stores design_info on fit so transform uses the same columns.
    """
    def __init__(self, formula: str):
        self.formula = formula
        self.design_info_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        dm = dmatrix(self.formula, X, return_type="dataframe")
        self.design_info_ = dm.design_info
        self.feature_names_ = dm.columns.to_list()
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        dm = build_design_matrices([self.design_info_], X)[0]
        return np.asarray(dm)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)
    
def assign_weighted_deciles(score: pd.Series, weight: pd.Series, n=10) -> pd.Series:
    """Return decile labels 1..n so each bucket has ~equal total weight."""
    s = pd.DataFrame({"score": score, "w": weight}).sort_values("score").reset_index()
    s["cw"] = s["w"].cumsum()
    total = s["w"].sum()
    # bins are based on cumulative weight share
    s["decile"] = np.ceil(n * s["cw"] / total).astype(int).clip(1, n)
    out = pd.Series(index=s["index"], data=s["decile"].values)
    return out.reindex(score.index)

def lift_table_by_decile(
    df: pd.DataFrame,
    y_pp_col: str,
    pred_pp: np.ndarray,
    exposure_col: str = "Exposure",
    n_deciles: int = 10,
):
    d = df.copy()
    d = d.loc[d[exposure_col] > 0].copy()

    d["pred_pp"] = pred_pp
    d["actual_loss"] = d[y_pp_col] * d[exposure_col]
    d["pred_loss"] = d["pred_pp"] * d[exposure_col]

    d["decile"] = assign_weighted_deciles(d["pred_pp"], d[exposure_col], n=n_deciles)

    g = (d.groupby("decile", as_index=False)
           .agg(
               exposure=(exposure_col, "sum"),
               actual_loss=("actual_loss", "sum"),
               pred_loss=("pred_loss", "sum"),
           ))

    g["actual_pp"] = g["actual_loss"] / g["exposure"]
    g["pred_pp"]   = g["pred_loss"]   / g["exposure"]

    port_actual_pp = d["actual_loss"].sum() / d[exposure_col].sum()
    port_pred_pp   = d["pred_loss"].sum()   / d[exposure_col].sum()

    g["actual_lift"] = g["actual_pp"] / port_actual_pp
    g["pred_lift"]   = g["pred_pp"]   / port_pred_pp

    return g, port_actual_pp, port_pred_pp

def plot_lift(g: pd.DataFrame, title="Lift Chart (Exposure-Weighted Deciles)"):
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

def gini_and_gain_curve_from_preds(
    df,
    pred_col,
    y_pp_col="pure_premium_capped_1MIL",
    exposure_col="Exposure"
):
    d = df.loc[df[exposure_col] > 0].copy()

    # use existing predictions
    d["pred_pp"] = d[pred_col]

    # convert to loss for a proper concentration/gain curve
    d["actual_loss"] = d[y_pp_col] * d[exposure_col]

    # sort highest predicted risk first
    d = d.sort_values("pred_pp", ascending=False).reset_index(drop=True)

    x = d[exposure_col].cumsum().to_numpy()
    y = d["actual_loss"].cumsum().to_numpy()

    x_share = np.r_[0.0, x / x[-1]]
    y_share = np.r_[0.0, y / y[-1]]

    auc = np.trapz(y_share, x_share)
    gini = 2 * auc - 1

    plt.figure()
    plt.plot(x_share, y_share, label=f"Gain curve (Gini={gini:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Diagonal")
    plt.xlabel("Cumulative exposure share")
    plt.ylabel("Cumulative actual loss share")
    plt.title("Gain Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return gini

def orderedLorenz(actual_losses, current_premium, new_premium, plot=True, sample_prop=0.1):
    dt = pd.DataFrame(
        {"y": actual_losses, "b": current_premium, "p": new_premium}
    ).dropna(subset=["y", "b", "p"])

    dt["r"] = dt["p"] / dt["b"]
    dt = dt.sort_values("r", kind="mergesort").reset_index(drop=True)

    dt["p_dist"] = dt["b"].cumsum() / dt["b"].sum()
    dt["l_dist"] = dt["y"].cumsum() / dt["y"].sum()

    # prepend a zero row (mirrors the R matrix(0, ...) trick)
    zero = pd.DataFrame([{col: 0 for col in dt.columns}])
    dt0 = pd.concat([zero, dt], ignore_index=True)

    # gini <- 1 - sum((l_dist + shift(l_dist)) * (p_dist - shift(p_dist)), na.rm=TRUE)
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
        if m <= 0:
            sample_idx = np.arange(n)
        else:
            sample_idx = np.random.choice(np.arange(n), size=m, replace=False)

        sample_dt = dt0.iloc[np.sort(sample_idx)]

        plt.figure()
        plt.plot(sample_dt["p_dist"], sample_dt["l_dist"])
        plt.plot([0, 1], [0, 1])
        plt.title(f"gini index = {gini:0.2f}")
        plt.ylabel("loss distribution")
        plt.xlabel("premium distribution")
        plt.show()

    return gini