from __future__ import annotations

import numpy as np

from src.compassign.rt.fast_ridge_prod import (
    RidgeGroupCompoundRTModel,
    bayesian_ridge_fit_from_sufficient_stats,
    ridge_fit_from_sufficient_stats,
)


def test_ridge_fit_from_sufficient_stats_recovers_linear_relation() -> None:
    rng = np.random.default_rng(42)
    n = 200
    d = 3
    x = rng.normal(size=(n, d))
    w_true = np.array([0.5, -1.2, 0.3])
    b_true = -0.7
    y = b_true + x @ w_true + rng.normal(scale=0.01, size=n)

    sum_x = x.sum(axis=0)[None, :]
    sum_y = np.array([y.sum()])
    sum_y2 = np.array([np.square(y).sum()])
    sum_xx = (x.T @ x)[None, :, :]
    sum_xy = (x.T @ y)[None, :]
    n_obs = np.array([n], dtype=np.int64)

    coefs, intercepts, sigma = ridge_fit_from_sufficient_stats(
        sum_x=sum_x,
        sum_y=sum_y,
        sum_y2=sum_y2,
        sum_xx=sum_xx,
        sum_xy=sum_xy,
        n=n_obs,
        lambda_ridge=1e-6,
    )

    assert coefs.shape == (1, d)
    assert intercepts.shape == (1,)
    assert sigma.shape == (1,)
    assert np.allclose(coefs[0], w_true, atol=5e-2)
    assert np.isfinite(intercepts[0])
    assert sigma[0] < 0.1


def test_ridge_model_predict_uses_fallback() -> None:
    feature_names = ("IS1", "RS4")
    # Only a single (super, comp) model exists: (1, 10)
    keys_super_comp = np.array([(1 << 32) + 10], dtype=np.int64)
    coefs_super_comp = np.array([[1.0, 0.0]])
    intercepts_super_comp = np.array([0.0])
    sigma_super_comp = np.array([0.1])

    # Compound-only fallback for comp_id=20
    comp_ids = np.array([20], dtype=np.int64)
    coefs_compound = np.array([[0.0, 1.0]])
    intercepts_compound = np.array([0.0])
    sigma_compound = np.array([0.2])

    model = RidgeGroupCompoundRTModel(
        feature_names=feature_names,
        keys_super_comp=keys_super_comp,
        coefs_super_comp=coefs_super_comp,
        intercepts_super_comp=intercepts_super_comp,
        sigma_super_comp=sigma_super_comp,
        comp_ids=comp_ids,
        coefs_compound=coefs_compound,
        intercepts_compound=intercepts_compound,
        sigma_compound=sigma_compound,
    )

    super_num = np.array([1, 2], dtype=np.int64)
    comp_id = np.array([10, 20], dtype=np.int64)
    x = np.array([[2.0, 3.0], [2.0, 3.0]])

    pred_mean, pred_std, used_fallback = model.predict(super_num=super_num, comp_id=comp_id, x=x)
    assert pred_mean.shape == (2,)
    assert pred_std.shape == (2,)
    assert used_fallback.shape == (2,)

    # (1,10) uses group-compound: pred = 1*IS1
    assert np.isclose(pred_mean[0], 2.0)
    assert np.isclose(pred_std[0], 0.1)
    assert not bool(used_fallback[0])

    # (2,20) missing group-compound, uses compound-only: pred = 1*RS4
    assert np.isclose(pred_mean[1], 3.0)
    assert np.isclose(pred_std[1], 0.2)
    assert bool(used_fallback[1])


def test_bayesian_ridge_fit_and_predict_posterior_shapes() -> None:
    rng = np.random.default_rng(42)
    n = 80
    d = 2
    x = rng.normal(size=(n, d))
    w_true = np.array([0.5, -1.2])
    b_true = 0.3
    y = b_true + x @ w_true + rng.normal(scale=0.05, size=n)

    sum_x = x.sum(axis=0)[None, :]
    sum_y = np.array([y.sum()])
    sum_y2 = np.array([np.square(y).sum()])
    sum_xx = (x.T @ x)[None, :, :]
    sum_xy = (x.T @ y)[None, :]
    n_obs = np.array([n], dtype=np.int64)

    a0 = 2.0
    (
        coefs,
        intercepts,
        sigma_resid,
        x_mean,
        v_diag,
        sigma2_mean,
        n_out,
    ) = bayesian_ridge_fit_from_sufficient_stats(
        sum_x=sum_x,
        sum_y=sum_y,
        sum_y2=sum_y2,
        sum_xx=sum_xx,
        sum_xy=sum_xy,
        n=n_obs,
        lambda_ridge=1e-3,
        a0=a0,
        b0=1e-6,
    )

    assert coefs.shape == (1, d)
    assert intercepts.shape == (1,)
    assert sigma_resid.shape == (1,)
    assert x_mean.shape == (1, d)
    assert v_diag.shape == (1, d)
    assert sigma2_mean.shape == (1,)
    assert n_out.shape == (1,)
    assert np.all(v_diag > 0)
    assert sigma2_mean[0] > 0

    feature_names = ("IS1", "RS4")
    model = RidgeGroupCompoundRTModel(
        feature_names=feature_names,
        keys_super_comp=np.array([(1 << 32) + 10], dtype=np.int64),
        coefs_super_comp=coefs,
        intercepts_super_comp=intercepts,
        sigma_super_comp=sigma_resid,
        comp_ids=np.zeros(0, dtype=np.int64),
        coefs_compound=np.zeros((0, d), dtype=float),
        intercepts_compound=np.zeros(0, dtype=float),
        sigma_compound=np.zeros(0, dtype=float),
        bayes_a0=a0,
        bayes_b0=1e-6,
        n_super_comp=n_out,
        x_mean_super_comp=x_mean,
        v_diag_super_comp=v_diag,
        sigma2_mean_super_comp=sigma2_mean,
        n_compound=np.zeros(0, dtype=np.int64),
        x_mean_compound=np.zeros((0, d), dtype=float),
        v_diag_compound=np.zeros((0, d), dtype=float),
        sigma2_mean_compound=np.zeros(0, dtype=float),
    )

    pred_mean, pred_std, pred_df, used_fallback = model.predict_posterior(
        super_num=np.array([1], dtype=np.int64),
        comp_id=np.array([10], dtype=np.int64),
        x=np.array([[0.1, -0.2]], dtype=float),
    )
    assert pred_mean.shape == (1,)
    assert pred_std.shape == (1,)
    assert pred_df.shape == (1,)
    assert used_fallback.shape == (1,)
    assert np.isfinite(pred_mean[0])
    assert pred_std[0] > 0
    assert np.isclose(pred_df[0], n + 2 * a0)
    assert not bool(used_fallback[0])
