"""Data preparation, modeling, diagnostics, and evaluation utilities.

The module prepares policy-level insurance data, creates exploratory plots,
fits OLS, Tweedie GLM, and XGBoost Tweedie models, and produces exposure-
weighted lift charts for held-out predictions.
"""

#######################
# Packages & Settings #
#######################
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb

from pathlib import Path
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedShuffleSplit

###############################
# Constants & Helper Functions #
###############################
CLAIM_CAPS = (
	1, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000,
	400_000, 500_000, 600_000, 700_000, 800_000, 900_000,
	1_000_000, 2_000_000,
)

CATEGORICAL_COLS = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
DEFAULT_TARGET_COL = "pure_premium_capped_1MIL"
DEFAULT_EXPOSURE_COL = "Exposure"
MISSING_CATEGORY = "__MISSING__"
FREQUENCY_DATA_ID = 41214
SEVERITY_DATA_ID = 41215


def fmt_cap(cap):
	"""Format a numeric claim cap as a compact column-name suffix.

	Args:
		cap: Maximum claim amount to format.

	Returns:
		A string such as ``"100K"`` or ``"1MIL"``.
	"""
	if cap >= 1_000_000:
		return f"{cap // 1_000_000}MIL"
	if cap >= 1_000:
		return f"{cap // 1_000}K"
	return str(cap)


def _safe_ratio(numerator, denominator):
	"""Divide arrays while returning zero wherever the denominator is zero.

	Args:
		numerator: Values to divide.
		denominator: Values by which to divide.

	Returns:
		A floating-point NumPy array containing the elementwise ratios.
	"""
	numerator = np.asarray(numerator, dtype=float)
	denominator = np.asarray(denominator, dtype=float)
	return np.divide(
		numerator,
		denominator,
		out=np.zeros_like(numerator),
		where=denominator != 0,
	)


#########################
# Import Data & Process #
#########################
def stratified_split_match_portfolio_freq(
	df,
	group_col="IDpol",
	exposure_col=DEFAULT_EXPOSURE_COL,
	claim_col="ClaimNb",
	test_size=0.20,
	q=10,
	tol=0.002,
	max_tries=300,
	random_state=42,
):
	"""Create train/test labels with closely matched portfolio frequency.

	Args:
	exposure_col=DEFAULT_EXPOSURE_COL,
		group_col: Column identifying a policy or other split unit.
		exposure_col: Exposure column used to calculate frequency.
		claim_col: Claim-count column used to calculate frequency.
		test_size: Fraction of policies assigned to the test set.
		q: Maximum number of frequency strata.
		tol: Maximum desired absolute train/test frequency difference.
		max_tries: Number of random seeds to try.
		random_state: First random seed to try.

	Returns:
		A copy of ``df`` with a ``set`` column containing ``train`` or ``test``.
	"""
	required = {group_col, exposure_col, claim_col}
	missing = required - set(df.columns)
	if missing:
		raise KeyError(f"Missing columns: {missing}")

	# Summarize each policy so all of its rows stay in the same split.
	policies = df.groupby(group_col).agg(
		Exposure_sum=(exposure_col, "sum"),
		ClaimNb_sum=(claim_col, "sum"),
	)
	policies = policies.loc[policies["Exposure_sum"] > 0].copy()
	if policies.empty:
		raise ValueError("No policies with positive exposure after aggregation.")
	policies["pol_freq"] = policies["ClaimNb_sum"] / policies["Exposure_sum"]

	# Create frequency strata, reducing the number of bins if small strata
	# would not contain at least one training and one test policy.
	q_try = min(q, policies["pol_freq"].nunique())
	bins = pd.qcut(policies["pol_freq"], q=q_try, duplicates="drop")
	while True:
		counts = bins.value_counts()
		if len(counts) >= 2 and (counts * test_size >= 1).all() and (
			counts * (1 - test_size) >= 1
		).all():
			break
		q_try -= 1
		if q_try < 2:
			bins = (policies["ClaimNb_sum"] > 0).astype(int)
			break
		bins = pd.qcut(policies["pol_freq"], q=q_try, duplicates="drop")

	labels = bins.astype(str)

	def portfolio_freq(split):
		"""Calculate exposure-weighted claim frequency for a policy summary."""
		return split["ClaimNb_sum"].sum() / split["Exposure_sum"].sum()

	best = None
	best_diff = float("inf")
	# Try nearby seeds and keep the split with the closest portfolio frequencies.
	for seed in range(random_state, random_state + max_tries):
		splitter = StratifiedShuffleSplit(
			n_splits=1, test_size=test_size, random_state=seed
		)
		train_idx, test_idx = next(splitter.split(np.zeros(len(policies)), labels))
		train = policies.iloc[train_idx]
		test = policies.iloc[test_idx]
		train_freq, test_freq = portfolio_freq(train), portfolio_freq(test)
		difference = abs(train_freq - test_freq)
		if difference < best_diff:
			best_diff = difference
			best = (train.index, test.index, train_freq, test_freq, seed)
			if difference <= tol:
				break

	if best is None:
		raise RuntimeError("Could not create a stratified train/test split.")

	train_policies, test_policies, train_freq, test_freq, used_seed = best
	result = df.copy()
	result["set"] = np.where(result[group_col].isin(test_policies), "test", "train")
	print(f"Used seed: {used_seed} | bins: {q_try if q_try >= 2 else 'has_claim'}")
	print(f"Overall PF: {portfolio_freq(policies):.6f}")
	print(f"Train PF  : {train_freq:.6f}")
	print(f"Test  PF  : {test_freq:.6f}")
	print(f"|Train-Test|: {best_diff:.6f} (tol={tol})")
	return result


def create_modeling_data(output_dir="data", raw_data_dir=None):
	"""Build, split, and save policy-level modeling data.

	Args:
		output_dir: Directory where raw, training, and test parquet files are saved.
		raw_data_dir: Optional directory containing cached raw OpenML parquet files.
			When omitted, ``output_dir`` is also used as the cache location.

	Returns:
		A tuple ``(df_train, df_test)`` containing policy-level modeling data.
	"""
	output_path = Path(output_dir)
	raw_path = Path(raw_data_dir) if raw_data_dir is not None else output_path
	raw_freq_path = raw_path / "df_raw_freq.parquet"
	raw_sev_path = raw_path / "df_raw_sev.parquet"

	# Use cached source data when available; otherwise retrieve both OpenML tables.
	if raw_freq_path.exists() and raw_sev_path.exists():
		df_raw_freq = pd.read_parquet(raw_freq_path)
		df_raw_sev = pd.read_parquet(raw_sev_path)
	elif raw_data_dir is not None:
		raise FileNotFoundError(f"Raw source Parquets not found in {raw_path}.")
	else:
		df_raw_freq = fetch_openml(data_id=FREQUENCY_DATA_ID, as_frame=True).frame
		df_raw_sev = fetch_openml(data_id=SEVERITY_DATA_ID, as_frame=True).frame

	if not df_raw_freq["IDpol"].is_unique:
		raise ValueError("The frequency data must contain one row per IDpol.")

	# Cap each claim before policy aggregation so large-loss assumptions can be
	# changed without losing the claim-level cap calculation.
	df_sev = df_raw_sev[["IDpol", "ClaimAmount"]].copy()
	claim_amount_cols = ["ClaimAmount"]
	for cap in CLAIM_CAPS:
		column = f"ClaimAmount_capped_{fmt_cap(cap)}"
		df_sev[column] = df_sev["ClaimAmount"].clip(upper=cap)
		claim_amount_cols.append(column)
	df_sev = df_sev.groupby("IDpol", as_index=False)[claim_amount_cols].sum()

	df = df_raw_freq.merge(df_sev, on="IDpol", how="left")
	claim_amount_cols = [
		column for column in df.columns if column.startswith("ClaimAmount")
	]
	df[claim_amount_cols] = df[claim_amount_cols].fillna(0)

	# Derive frequency, severity, and pure-premium targets while avoiding
	# undefined ratios for policies with zero exposure or zero claims.
	df["frequency"] = _safe_ratio(df["ClaimNb"], df["Exposure"])
	df["severity_uncapped"] = _safe_ratio(df["ClaimAmount"], df["ClaimNb"])
	for cap in CLAIM_CAPS:
		label = fmt_cap(cap)
		amount_col = f"ClaimAmount_capped_{label}"
		df[f"severity_capped_{label}"] = _safe_ratio(df[amount_col], df["ClaimNb"])

	df["pure_premium_uncapped"] = _safe_ratio(df["ClaimAmount"], df["Exposure"])
	for cap in CLAIM_CAPS:
		label = fmt_cap(cap)
		amount_col = f"ClaimAmount_capped_{label}"
		df[f"pure_premium_capped_{label}"] = _safe_ratio(df[amount_col], df["Exposure"])

	df = stratified_split_match_portfolio_freq(df)
	df_train = df.loc[df["set"] == "train"].copy()
	df_test = df.loc[df["set"] == "test"].copy()

	output_path.mkdir(parents=True, exist_ok=True)
	df_raw_freq.to_parquet(output_path / "df_raw_freq.parquet")
	df_raw_sev.to_parquet(output_path / "df_raw_sev.parquet")
	df_train.to_parquet(output_path / "df_train.parquet")
	df_test.to_parquet(output_path / "df_test.parquet")
	print(f"Saved {len(df_train):,} training and {len(df_test):,} test policies to {output_path}.")
	return df_train, df_test


###############################
# Exploratory Data Analysis  #
###############################
def plot_pure_premium_distribution(
	df, target_col="pure_premium_uncapped", bins=50, view_exposure_share=0.95
):
	"""Plot zero versus positive pure premium and its weighted positive tail.

	df["pure_premium_uncapped"] = _safe_ratio(df["ClaimAmount"], df[DEFAULT_EXPOSURE_COL])
		df: Policy-level data containing the target and exposure columns.
		target_col: Pure-premium column to visualize.
		bins: Number of histogram bins for positive pure premium.
		view_exposure_share: Exposure share retained in the focused histogram view.

	Returns:
		The Matplotlib figure containing both distribution plots.
	"""
	valid = df[target_col].notna() & df["Exposure"].gt(0)
	data = df.loc[valid, [target_col, "Exposure"]]
	zero_exposure = data.loc[data[target_col].eq(0), "Exposure"].sum()
	positive_exposure = data.loc[data[target_col].gt(0), "Exposure"].sum()
	total_exposure = zero_exposure + positive_exposure
	if total_exposure <= 0:
		raise ValueError("No positive exposure found for the target variable.")

	shares = 100 * np.array([zero_exposure, positive_exposure]) / total_exposure
	positive = data.loc[data[target_col] > 0]
	if positive.empty:
		raise ValueError(f"No positive-exposure values found for {target_col}.")
	values = positive[target_col].to_numpy(dtype=float)
	weights = positive["Exposure"].to_numpy(dtype=float)
	# Find a display range containing most positive exposure while retaining
	# the high-loss tail as a labeled share of exposure.
	order = np.argsort(values)
	cumulative_exposure = np.cumsum(weights[order])
	cutoff_index = np.searchsorted(
		cumulative_exposure, view_exposure_share * cumulative_exposure[-1]
	)
	view_max = values[order][min(cutoff_index, len(values) - 1)]
	in_view = values <= view_max
	bin_edges = np.linspace(0, view_max, bins + 1)
	shown_exposure = weights[in_view].sum()
	tail_share = 1 - shown_exposure / weights.sum()

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
	bars = axes[0].bar(
		["Target = $0", "Target > $0"],
		shares,
		color=["#C45A32", "#167D8D"],
		width=0.6,
	)
	axes[0].bar_label(bars, labels=[f"{share:.1f}%" for share in shares], padding=4)
	axes[0].set_ylim(0, 108)
	axes[0].set_ylabel("Share of total exposure (%)")
	axes[0].set_title("Exposure by pure-premium outcome")
	axes[0].grid(axis="y", alpha=0.25)
	axes[0].set_axisbelow(True)

	axes[1].hist(
		values[in_view], bins=bin_edges, weights=weights[in_view],
		color="#167D8D", edgecolor="white"
	)
	axes[1].set_xlabel("Uncapped pure premium ($, untransformed)")
	axes[1].set_ylabel("Exposure (years)")
	axes[1].set_title("Positive pure premium, weighted by exposure")
	axes[1].set_xlim(0, view_max)
	axes[1].text(
		0.98,
		0.95,
		f"{tail_share:.1%} of positive exposure above ${view_max:,.0f}",
		transform=axes[1].transAxes,
		ha="right",
		va="top",
	)
	axes[1].grid(axis="y", alpha=0.25)
	axes[1].set_axisbelow(True)
	fig.tight_layout()
	return fig


def plot_one_way_pure_premium(df, dimension, bins=12):
	"""Plot exposure and pure premium by one rating-factor dimension.

	Args:
		df: Policy-level data containing ``dimension``, ``Exposure``, and
			``ClaimAmount``.
		dimension: Categorical or numeric feature to group or quantile-bin.
		bins: Maximum number of quantile bins for numeric dimensions.

	Returns:
		The Matplotlib figure containing exposure bars and pure-premium estimates.
	"""
	data = df[[dimension, "Exposure", "ClaimAmount"]].copy()
	data["_row_pure_premium"] = np.divide(
		data["ClaimAmount"],
		data["Exposure"],
		out=np.full(len(data), np.nan, dtype=float),
		where=data["Exposure"].to_numpy(dtype=float) != 0,
	)
	if is_numeric_dtype(data[dimension]) and data[dimension].nunique() > 1:
		# Quantile bins keep numeric groups reasonably populated for the plot.
		data["_group"] = pd.qcut(
			data[dimension], q=min(bins, data[dimension].nunique()), duplicates="drop"
		)
	else:
		data["_group"] = data[dimension]
	grouped = data.groupby("_group", dropna=False, observed=True, sort=True).agg(
		Exposure=("Exposure", "sum"),
		ClaimAmount=("ClaimAmount", "sum"),
		RowStd=("_row_pure_premium", "std"),
		RowCount=("_row_pure_premium", "count"),
	)
	# Use total losses divided by total exposure for the group estimate;
	# estimate uncertainty from variation in policy-level pure premiums.
	grouped["PurePremium"] = _safe_ratio(grouped["ClaimAmount"], grouped["Exposure"])
	grouped["SE"] = grouped["RowStd"] / np.sqrt(grouped["RowCount"].clip(lower=1))
	portfolio = _safe_ratio(data["ClaimAmount"].sum(), data["Exposure"].sum())

	positions = np.arange(len(grouped))
	labels = grouped.index.astype(str)
	fig, exposure_ax = plt.subplots(figsize=(12, 6))
	bars = exposure_ax.bar(
		positions, grouped["Exposure"], color="#9BB7B5", alpha=0.75, label="Exposure"
	)
	exposure_ax.bar_label(bars, fmt="%.0f")
	exposure_ax.set_xticks(positions)
	exposure_ax.set_xticklabels(labels, rotation=35 if is_numeric_dtype(data[dimension]) else 0, ha="right" if is_numeric_dtype(data[dimension]) else "center")
	exposure_ax.set_xlabel(dimension)
	exposure_ax.set_ylabel("Exposure (years)")

	premium_ax = exposure_ax.twinx()
	line, = premium_ax.plot(
		positions, grouped["PurePremium"], color="#C45A32", marker="o", label="Pure premium"
	)
	premium_ax.fill_between(
		positions,
		(grouped["PurePremium"] - grouped["SE"]).clip(lower=0),
		grouped["PurePremium"] + grouped["SE"],
		color="#C45A32",
		alpha=0.18,
		label="±1 SE",
	)
	portfolio_line = premium_ax.axhline(
		portfolio, color="#315D6B", linestyle="--", label="Portfolio"
	)
	premium_ax.set_ylabel("Uncapped pure premium ($)")
	premium_ax.set_title(f"Uncapped pure premium and exposure by {dimension}")
	exposure_ax.legend(
		[bars, line, portfolio_line], ["Exposure", "Pure premium", "Portfolio"],
		loc="upper left",
	)
	fig.tight_layout()
	return fig


def plot_area_one_way(df):
	"""Plot the one-way pure-premium relationship for ``Area``.

	Args:
		df: Policy-level modeling data.

	Returns:
		The Matplotlib figure for the Area analysis.
	"""
	return plot_one_way_pure_premium(df, "Area")


def plot_bonus_malus_one_way(df):
	"""Plot the one-way pure-premium relationship for ``BonusMalus``.

	Args:
		df: Policy-level modeling data.

	Returns:
		The Matplotlib figure for the BonusMalus analysis.
	"""
	return plot_one_way_pure_premium(df, "BonusMalus")


def _build_model_formula(target_col):
	"""Build the shared formula used by the OLS and Tweedie GLM models.

	Args:
		target_col: Response column to place on the left side of the formula.

	Returns:
		A Patsy formula with categorical indicators and numeric rating features.
	"""
	predictors = [f"C({column})" for column in CATEGORICAL_COLS] + NUMERIC_COLS
	return f"{target_col} ~ {' + '.join(predictors)}"


def _prepare_xgb_features(df, categories=None):
	"""Prepare categorical and numeric columns for an XGBoost matrix.

	Args:
		df: DataFrame containing the shared rating features.
		categories: Optional mapping of training categorical levels. When supplied,
			new data is aligned to those levels before prediction.

	Returns:
		A tuple containing the prepared feature DataFrame and its categorical levels.
	"""
	X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
	for column in CATEGORICAL_COLS:
		X[column] = X[column].astype("category")
		if categories is None:
			if X[column].isna().any():
				X[column] = X[column].cat.add_categories([MISSING_CATEGORY]).fillna(MISSING_CATEGORY)
		else:
			X[column] = X[column].cat.set_categories(categories[column])
			if MISSING_CATEGORY in categories[column]:
				X[column] = X[column].fillna(MISSING_CATEGORY)

	for column in NUMERIC_COLS:
		X[column] = pd.to_numeric(X[column], errors="coerce")
	X[NUMERIC_COLS] = X[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	if categories is None:
		categories = {column: X[column].cat.categories for column in CATEGORICAL_COLS}
	return X, categories


##########################################
# Modeling: Ordinary Least Squares (OLS) #
##########################################
def fit_ols_model(df_train, target_col=DEFAULT_TARGET_COL):
	"""Fit an OLS pure-premium model using categorical and numeric rating features.

	Args:
		df_train: Training data containing the target and rating features.
		target_col: Response column to model.

	Returns:
		A fitted statsmodels OLS results object.
	"""
	formula = _build_model_formula(target_col)
	return smf.ols(formula=formula, data=df_train).fit()


############################################
# Modeling: Generalized Linear Model (GLM) #
############################################
def fit_tweedie_glm(df_train, target_col=DEFAULT_TARGET_COL, variance_power=1.85):
	"""Fit a Tweedie GLM with a log link for pure premium.

	Args:
		df_train: Training data containing the target and rating features.
		target_col: Response column to model.
		variance_power: Tweedie variance power, where values between 1 and 2
			model a compound Poisson-Gamma response.

	Returns:
		A fitted statsmodels GLM results object.
	"""
	# The log link keeps fitted mean pure premiums positive, while Tweedie
	# variance power 1 < p < 2 accommodates zeros and a positive skewed tail.
	formula = _build_model_formula(target_col)
	family = sm.families.Tweedie(
		var_power=variance_power,
		link=sm.families.links.Log(),
	)
	return smf.glm(formula=formula, data=df_train, family=family).fit()

##############################
# Modeling: OLS Diagnostics #
##############################
def plot_ols_residual_qq(ols_result):
	"""Plot a normal Q-Q diagnostic for standardized OLS residuals.

	Args:
		ols_result: Fitted statsmodels OLS results object.

	Returns:
		The Matplotlib figure containing the Q-Q plot.
	"""
	# Standardization makes residuals comparable across observations with
	# different fitted-value uncertainty.
	standardized_residuals = ols_result.get_influence().resid_studentized_internal
	fig, ax = plt.subplots(figsize=(7, 7))
	sm.qqplot(standardized_residuals, line="45", ax=ax)
	ax.set_title("Normal Q-Q Plot of OLS Residuals")
	fig.tight_layout()
	return fig


def lowest_negative_ols_fits(ols_result, n=10):
	"""Return the most negative fitted values from an OLS result.

	Args:
		ols_result: Fitted statsmodels OLS results object.
		n: Maximum number of negative fitted values to return.

	Returns:
		A DataFrame containing the ``n`` smallest negative predictions.
	"""
	# Negative loss predictions are impossible, so surface the worst cases.
	negative_fits = ols_result.fittedvalues.loc[ols_result.fittedvalues < 0]
	return negative_fits.nsmallest(n).rename("Fitted pure premium").to_frame()


####################################
# Modeling: Gradient Boosting (GBM) #
####################################
def tune_and_fit_xgb_tweedie(
	df_train,
	target_col=DEFAULT_TARGET_COL,
	exposure_col=DEFAULT_EXPOSURE_COL,
	n_splits=5,
	n_trials=25,
	random_state=42,
	num_boost_round_max=5000,
	early_stopping_rounds=50,
):
	"""Tune and fit an exposure-weighted XGBoost Tweedie model.

	Args:
		df_train: Training data containing the target, exposure, and features.
		target_col: Response column to model.
		exposure_col: Exposure column used as the observation weight.
		n_splits: Number of cross-validation folds.
		n_trials: Number of Optuna hyperparameter trials.
		random_state: Seed used for cross-validation and XGBoost.
		num_boost_round_max: Maximum boosting rounds considered during CV.
		early_stopping_rounds: CV rounds without improvement before stopping.

	Returns:
		A tuple containing the fitted booster, Optuna study, selected boosting
		rounds, and training categorical levels.
	"""
	# Match XGBoost's native categorical representation and normalize numeric
	# inputs before creating the training matrix.
	X, train_categories = _prepare_xgb_features(df_train)

	y = pd.to_numeric(df_train[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
	weights = pd.to_numeric(df_train[exposure_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
	weighted_mean = np.sum(weights * y) / max(np.sum(weights), 1e-12)
	# XGBoost's Tweedie objective works on the log scale; initialize it at the
	# exposure-weighted mean pure premium to give optimization a sensible start.
	base_score = float(np.log(weighted_mean + 1e-12))
	dtrain = xgb.DMatrix(X, label=y, weight=weights, enable_categorical=True)
	kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	folds = [(train_idx, valid_idx) for train_idx, valid_idx in kfold.split(X)]

	def objective(trial):
		"""Evaluate one Optuna hyperparameter configuration with CV."""
		# Each trial tests one Tweedie power and one tree configuration using
		# the same folds, so the trial scores are directly comparable.
		variance_power = trial.suggest_float("tweedie_variance_power", 1.2, 1.9, step=0.05)
		params = {
			"objective": "reg:tweedie",
			"tweedie_variance_power": variance_power,
			"eval_metric": f"tweedie-nloglik@{variance_power:.2f}",
			"base_score": base_score,
			"tree_method": "hist",
			"device": "cuda",
			"seed": random_state,
			"eta": trial.suggest_float("eta", 0.01, 0.15, step=0.01),
			"max_depth": trial.suggest_int("max_depth", 2, 10),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 100_000),
			"gamma": trial.suggest_float("gamma", 0.0, 10.0, step=0.5),
			"subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),
			"lambda": trial.suggest_float("lambda", 0.0, 50.0, step=1.0),
			"alpha": trial.suggest_float("alpha", 0.0, 10.0, step=0.5),
			"max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0, step=0.5),
			"max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 16),
			"max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 256),
		}
		cv_results = xgb.cv(
			params=params,
			dtrain=dtrain,
			num_boost_round=num_boost_round_max,
			folds=folds,
			early_stopping_rounds=early_stopping_rounds,
			verbose_eval=False,
		)
		metric_col = next(
			column for column in cv_results.columns
			if column.startswith("test-") and column.endswith("-mean")
		)
		return float(cv_results[metric_col].min())

	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=n_trials)
	best_trial = study.best_trial
	# Refit cross-validation with the winning parameters to select the number
	# of boosting rounds before training on all available training data.
	best_params = {
		"objective": "reg:tweedie",
		"eval_metric": f"tweedie-nloglik@{best_trial.params['tweedie_variance_power']:.2f}",
		"base_score": base_score,
		"tree_method": "hist",
		"device": "cuda",
		"seed": random_state,
	}
	best_cv = xgb.cv(
		params=best_params,
		dtrain=dtrain,
		num_boost_round=num_boost_round_max,
		folds=folds,
		early_stopping_rounds=early_stopping_rounds,
		verbose_eval=False,
	)
	best_num_boost_round = int(best_cv.shape[0])
	booster = xgb.train(
		params=best_params,
		dtrain=dtrain,
		num_boost_round=best_num_boost_round,
	)
	print(f"Best CV Tweedie negative log-likelihood: {study.best_value:.6f}")
	print(f"Best boosting rounds: {best_num_boost_round}")
	print(f"Best parameters: {best_params}")
	return booster, study, best_num_boost_round, train_categories


####################################################
# Model Comparisons: Exposure-Weighted Lift Charts #
####################################################
def _lift_table_by_exposure_decile(df, prediction, target_col, exposure_col, n_deciles=10):
	"""Calculate actual and predicted lift by exposure-weighted decile.

	Args:
		df: Evaluation data containing target and exposure columns.
		prediction: Predicted pure-premium values aligned with ``df``.
		target_col: Actual pure-premium column.
		exposure_col: Exposure column used to form weighted deciles.
		n_deciles: Number of prediction groups to create.

	Returns:
		A DataFrame containing actual, predicted, and relative lift by decile.
	"""
	data = df[[target_col, exposure_col]].copy()
	data["prediction"] = np.asarray(prediction)
	data = data.replace([np.inf, -np.inf], np.nan).dropna()
	data = data.loc[data[exposure_col] > 0].sort_values("prediction", kind="mergesort")
	if data.empty:
		raise ValueError("No finite test observations with positive exposure remain.")

	total_exposure = data[exposure_col].sum()
	actual_portfolio = np.average(data[target_col], weights=data[exposure_col])
	predicted_portfolio = np.average(data["prediction"], weights=data[exposure_col])
	if actual_portfolio <= 0 or predicted_portfolio <= 0:
		raise ValueError("Portfolio actual and predicted pure premiums must be positive.")

	cumulative_exposure = data[exposure_col].cumsum()
	# Sorting by prediction and using cumulative exposure makes each group carry
	# approximately the same amount of exposure rather than the same row count.
	data["decile"] = np.floor(
		(cumulative_exposure - 1e-12) / total_exposure * n_deciles
	).astype(int).add(1).clip(1, n_deciles)

	lift = data.groupby("decile", as_index=False).apply(
		lambda group: pd.Series({
			"actual_pp": np.average(group[target_col], weights=group[exposure_col]),
			"predicted_pp": np.average(group["prediction"], weights=group[exposure_col]),
		}),
		include_groups=False,
	).reset_index(drop=True)
	lift["actual_lift"] = lift["actual_pp"] / actual_portfolio
	lift["predicted_lift"] = lift["predicted_pp"] / predicted_portfolio
	return lift


def plot_test_model_lifts(
	df_test,
	ols_result,
	glm_result,
	gbm_model,
	gbm_categories,
	target_col=DEFAULT_TARGET_COL,
	exposure_col=DEFAULT_EXPOSURE_COL,
):
	"""Plot held-out lift charts for OLS, GLM, and GBM predictions.

	Args:
		df_test: Held-out policy-level data.
		ols_result: Fitted statsmodels OLS results object.
		glm_result: Fitted statsmodels Tweedie GLM results object.
		gbm_model: Fitted XGBoost booster.
		gbm_categories: Training categorical levels used by the GBM.
		target_col: Actual pure-premium column used for evaluation.
		exposure_col: Exposure column used to weight the lift deciles.

	Returns:
		None. Displays one lift chart for each model.
	"""
	# Reuse the training category levels so test-set columns have the exact
	# representation expected by the fitted XGBoost model.
	X_test, _ = _prepare_xgb_features(df_test, categories=gbm_categories)
	gbm_matrix = xgb.DMatrix(X_test, enable_categorical=True)

	predictions = {
		"OLS": ols_result.predict(df_test),
		"Tweedie GLM": glm_result.predict(df_test),
		"GBM": gbm_model.predict(gbm_matrix),
	}
	# Apply the same ranking and exposure weighting to every model so the charts
	# compare discrimination on an identical held-out portfolio.
	for model_name, prediction in predictions.items():
		lift = _lift_table_by_exposure_decile(
			df_test, prediction, target_col, exposure_col
		)
		fig, ax = plt.subplots(figsize=(8, 4.5))
		ax.plot(lift["decile"], lift["actual_lift"], marker="o", label="Actual pure premium")
		ax.plot(lift["decile"], lift["predicted_lift"], marker="o", label="Predicted pure premium")
		ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
		ax.set(
			xlabel="Prediction decile (1 = lowest predicted risk, 10 = highest)",
			ylabel="Pure premium (multiple of test-set average)",
			title=f"{model_name}: pure premium by test-set exposure decile",
		xlim=(1, 10),
		)
		ax.set_xticks(range(1, 11))
		ax.grid(axis="y", alpha=0.25)
		ax.legend()
		fig.tight_layout()
		plt.show()
		plt.close(fig)
