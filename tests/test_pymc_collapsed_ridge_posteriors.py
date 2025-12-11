from __future__ import annotations

import numpy as np

from src.compassign.rt import pymc_collapsed_ridge as cr


def _center_suffstats(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x.mean(axis=0)
    ym = float(y.mean())
    xc = x - xm[None, :]
    yc = y - ym
    xtx = xc.T @ xc
    xty = xc.T @ yc
    return xm, ym, xtx, xty


def test_b_prior_posterior_matches_dense_formula() -> None:
    rng = np.random.default_rng(42)
    n = 9
    p = 3
    x = rng.normal(size=(n, p))

    beta_true = np.asarray([2.0, 0.3, -0.2, 0.1], dtype=np.float64)
    y = beta_true[0] + x @ beta_true[1:] + rng.normal(scale=0.05, size=n)

    xm, ym, xtx, xty = _center_suffstats(x, y)
    xtx_arr = xtx[None, :, :]
    xty_arr = xty[None, :]
    x_mean_arr = xm[None, :]
    y_mean_arr = np.asarray([ym], dtype=np.float64)
    n_arr = np.asarray([n], dtype=np.int64)

    group_cluster_idx = np.asarray([0], dtype=np.int64)
    group_chem_idx = np.asarray([0], dtype=np.int64)
    mu_cluster = np.asarray([0.5], dtype=np.float64)
    t_chem = np.asarray([1.2], dtype=np.float64)
    tau_b = 0.7
    lambda_slopes = 2.0
    sigma2 = 0.25

    lambda_diag = np.full((p,), float(lambda_slopes), dtype=np.float64)

    beta_hat, beta_var_diag, beta_cov = cr._compute_group_posterior_summaries_with_b_prior(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        group_cluster_idx=group_cluster_idx,
        group_chem_idx=group_chem_idx,
        mu_cluster=mu_cluster,
        t_chem=t_chem,
        tau_b=tau_b,
        lambda_diag=lambda_diag,
        sigma2_mean=sigma2,
    )

    x1 = np.concatenate([np.ones((n, 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    prior_mean = np.concatenate(
        [[float(mu_cluster[0] + t_chem[0])], np.zeros((p,), dtype=np.float64)]
    )
    prior_prec = np.diag([1.0 / (tau_b**2), *([lambda_slopes / sigma2] * p)]).astype(np.float64)
    post_prec = (x1.T @ x1) / sigma2 + prior_prec
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ ((x1.T @ y) / sigma2 + prior_prec @ prior_mean)

    assert np.allclose(beta_hat[0], post_mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(beta_cov[0], post_cov, rtol=1e-10, atol=1e-10)
    assert np.allclose(beta_var_diag[0], np.diagonal(post_cov), rtol=1e-10, atol=1e-10)


def test_b_prior_with_slope_mean_matches_dense_formula() -> None:
    rng = np.random.default_rng(7)
    n = 10
    p = 3
    x = rng.normal(size=(n, p))

    beta_true = np.asarray([0.7, 0.15, -0.1, 0.05], dtype=np.float64)
    y = beta_true[0] + x @ beta_true[1:] + rng.normal(scale=0.03, size=n)

    xm, ym, xtx, xty = _center_suffstats(x, y)
    xtx_arr = xtx[None, :, :]
    xty_arr = xty[None, :]
    x_mean_arr = xm[None, :]
    y_mean_arr = np.asarray([ym], dtype=np.float64)
    n_arr = np.asarray([n], dtype=np.int64)

    group_cluster_idx = np.asarray([0], dtype=np.int64)
    group_chem_idx = np.asarray([0], dtype=np.int64)
    mu_cluster = np.asarray([0.2], dtype=np.float64)
    t_chem = np.asarray([0.1], dtype=np.float64)
    tau_b = 0.6
    lambda_slopes = 1.7
    sigma2 = 0.12

    slope_mean_cluster = np.asarray([[0.4, -0.2, 0.1]], dtype=np.float64)
    lambda_diag = np.full((p,), float(lambda_slopes), dtype=np.float64)

    beta_hat, beta_var_diag, beta_cov = cr._compute_group_posterior_summaries_with_b_prior(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        group_cluster_idx=group_cluster_idx,
        group_chem_idx=group_chem_idx,
        mu_cluster=mu_cluster,
        t_chem=t_chem,
        slope_mean_cluster=slope_mean_cluster,
        tau_b=tau_b,
        lambda_diag=lambda_diag,
        sigma2_mean=sigma2,
    )

    x1 = np.concatenate([np.ones((n, 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    prior_mean = np.concatenate(
        [[float(mu_cluster[0] + t_chem[0])], slope_mean_cluster[0].astype(np.float64)]
    )
    prior_prec = np.diag([1.0 / (tau_b**2), *([lambda_slopes / sigma2] * p)]).astype(np.float64)
    post_prec = (x1.T @ x1) / sigma2 + prior_prec
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ ((x1.T @ y) / sigma2 + prior_prec @ prior_mean)

    assert np.allclose(beta_hat[0], post_mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(beta_cov[0], post_cov, rtol=1e-10, atol=1e-10)
    assert np.allclose(beta_var_diag[0], np.diagonal(post_cov), rtol=1e-10, atol=1e-10)


def test_b_prior_diffuse_matches_flat_intercept() -> None:
    rng = np.random.default_rng(123)
    n = 12
    p = 2
    x = rng.normal(size=(n, p))
    beta_true = np.asarray([1.5, 0.7, -0.4], dtype=np.float64)
    y = beta_true[0] + x @ beta_true[1:]

    xm, ym, xtx, xty = _center_suffstats(x, y)
    xtx_arr = xtx[None, :, :]
    xty_arr = xty[None, :]
    x_mean_arr = xm[None, :]
    y_mean_arr = np.asarray([ym], dtype=np.float64)
    n_arr = np.asarray([n], dtype=np.int64)

    lambda_slopes = 1.3
    sigma2 = 0.2

    lambda_diag = np.full((p,), float(lambda_slopes), dtype=np.float64)

    beta_hat_flat, _, beta_cov_flat = cr._compute_group_posterior_summaries(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        lambda_diag=lambda_diag,
        sigma2_mean=sigma2,
    )

    beta_hat_bprior, _, beta_cov_bprior = cr._compute_group_posterior_summaries_with_b_prior(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        group_cluster_idx=np.asarray([0], dtype=np.int64),
        group_chem_idx=np.asarray([0], dtype=np.int64),
        mu_cluster=np.asarray([0.0], dtype=np.float64),
        t_chem=np.asarray([0.0], dtype=np.float64),
        tau_b=1e9,
        lambda_diag=lambda_diag,
        sigma2_mean=sigma2,
    )

    assert np.allclose(beta_hat_bprior, beta_hat_flat, rtol=1e-8, atol=1e-10)
    assert np.allclose(beta_cov_bprior, beta_cov_flat, rtol=1e-8, atol=1e-10)


def test_flat_intercept_with_slope_mean_matches_dense_formula() -> None:
    rng = np.random.default_rng(202)
    n = 11
    p = 3
    x = rng.normal(size=(n, p))

    beta_true = np.asarray([1.1, 0.2, -0.15, 0.05], dtype=np.float64)
    y = beta_true[0] + x @ beta_true[1:] + rng.normal(scale=0.04, size=n)

    xm, ym, xtx, xty = _center_suffstats(x, y)
    xtx_arr = xtx[None, :, :]
    xty_arr = xty[None, :]
    x_mean_arr = xm[None, :]
    y_mean_arr = np.asarray([ym], dtype=np.float64)
    n_arr = np.asarray([n], dtype=np.int64)

    lambda_slopes = 1.9
    sigma2 = 0.3
    slope_mean = np.asarray([[0.4, -0.2, 0.1]], dtype=np.float64)
    lambda_diag = np.full((p,), float(lambda_slopes), dtype=np.float64)

    beta_hat, beta_var_diag, beta_cov = cr._compute_group_posterior_summaries(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        lambda_diag=lambda_diag,
        slope_mean=slope_mean,
        sigma2_mean=sigma2,
    )

    x1 = np.concatenate([np.ones((n, 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    prior_mean = np.concatenate([[0.0], slope_mean[0].astype(np.float64)])
    prior_prec = np.diag([0.0, *([lambda_slopes / sigma2] * p)]).astype(np.float64)
    post_prec = (x1.T @ x1) / sigma2 + prior_prec
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ ((x1.T @ y) / sigma2 + prior_prec @ prior_mean)

    assert np.allclose(beta_hat[0], post_mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(beta_cov[0], post_cov, rtol=1e-10, atol=1e-10)
    assert np.allclose(beta_var_diag[0], np.diagonal(post_cov), rtol=1e-10, atol=1e-10)
