#!/usr/bin/env python
"""Reproducible analysis of the verified product-application coding."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "real_data_analysis_output"

content = pd.read_csv(OUT / "content_coding.csv")
quality = pd.read_csv(OUT / "production_quality.csv")
data = content.merge(quality[["filename", "n_cuts", "cut_rate_per_s"]], on="filename")


def contingency(frame):
    viral = frame.loc[frame.condition.eq("viral"), "product_applied"].astype(int)
    flop = frame.loc[frame.condition.eq("flop"), "product_applied"].astype(int)
    table = [[int(viral.sum()), int((1 - viral).sum())],
             [int(flop.sum()), int((1 - flop).sum())]]
    test = stats.fisher_exact(table)
    a, b = table[0]
    c, d = table[1]
    odds_ratio = (a * d) / (b * c)
    se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci = np.exp(np.log(odds_ratio) + np.array([-1, 1]) * 1.96 * se_log)
    return table, float(test.statistic), float(test.pvalue), ci


rows = []
for label, frame in (("all advertisements", data),
                     ("makeup advertisements", data[data.category.str.lower().eq("makeup")])):
    table, odds_ratio, p, ci = contingency(frame)
    rows.append({"analysis": label, "viral_yes": table[0][0], "viral_no": table[0][1],
                 "flop_yes": table[1][0], "flop_no": table[1][1],
                 "odds_ratio": odds_ratio, "ci95_low": ci[0], "ci95_high": ci[1],
                 "p": p})

pd.DataFrame(rows).to_csv(OUT / "product_application_verified_fisher.csv", index=False)

correlations = []
for predictor in ("n_cuts", "cut_rate_per_s"):
    r, p = stats.pearsonr(data.product_applied, data[predictor])
    correlations.append({"predictor": predictor, "r": r, "p": p})
pd.DataFrame(correlations).to_csv(OUT / "product_application_verified_correlations.csv", index=False)

# Unpenalised clip-level logistic regression: viral status ~ product application + cut rate.
y = data.condition.eq("viral").astype(float).to_numpy()
X = np.column_stack([np.ones(len(data)), data.product_applied, data.cut_rate_per_s])


def negative_log_likelihood(beta):
    probability = expit(X @ beta)
    return -np.sum(y * np.log(probability + 1e-15) +
                   (1 - y) * np.log(1 - probability + 1e-15))


fit = minimize(negative_log_likelihood, np.zeros(X.shape[1]), method="BFGS")
beta = fit.x
probability = expit(X @ beta)
covariance = np.linalg.inv(X.T @ ((probability * (1 - probability))[:, None] * X))
se = np.sqrt(np.diag(covariance))
p_values = 2 * stats.norm.sf(np.abs(beta / se))
pd.DataFrame({
    "term": ["intercept", "product_applied", "cut_rate_per_s"],
    "b": beta,
    "se": se,
    "p": p_values,
    "ci95_low": beta - 1.96 * se,
    "ci95_high": beta + 1.96 * se,
}).to_csv(OUT / "product_application_verified_logistic.csv", index=False)

print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(correlations).to_string(index=False))
print("Logistic convergence:", fit.success, fit.message)
