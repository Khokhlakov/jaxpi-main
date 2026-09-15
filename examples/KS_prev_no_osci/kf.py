"""
Kalman filtering for the KS DeepONet surrogate.

This is the KS translation of ``examples.l96_f.kf``. The filtering
algebra is dimension-agnostic, so the three inflation schemes carry over
unchanged in substance:

    make_enkf            -- multiplicative covariance inflation
    make_route_b_enkf    -- residual-scaled (physics-driven) additive
                            inflation:  Q_n^i = (alpha + beta*||rho_n^i||^2) Q0
    make_rtpp_enkf       -- relaxation-to-prior perturbations

together with their fine-step / window-aware smoothers
(``run_enkf_smoother``, ``run_enkf_smoother_route_b``,
``run_enkf_smoother_rtpp``) and ``init_ensemble``.

What changes relative to L96
----------------------------
1. **No augmented parameter.** The L96 filter carries a 41-D state --
   40 dynamic variables plus the per-trajectory forcing F, which is
   propagated but never inflated (hence the ``N_dyn`` argument and the
   inflation/RTPP masks in the L96 code). The KS surrogate's state is
   just the 1-D field sampled on the solver's N-point periodic grid, so
   by default every component is dynamic and every mask is uniform.
   ``N_dyn`` is retained as an optional argument (defaulting to ``N``)
   so a future parameter-augmented KS variant -- estimating, say, the
   domain length or a forcing amplitude alongside the field -- can reuse
   these factories without modification.

2. **The state is a FIELD, not a list of variables.** Neighbouring
   entries of the KS state vector are neighbouring points of a smooth
   spatial profile, so a diagonal P0/Q0 (independent white noise per
   grid point) generates ensemble perturbations dominated by
   grid-scale wavenumbers -- precisely the modes the KS operator
   (-u_xx - u_xxxx) damps hardest, so they decay almost immediately and
   contribute little usable spread. ``periodic_gaussian_cov`` builds a
   smooth, spatially correlated alternative; see its docstring.

3. **The EKF is kept.** ``models.KSUDON.make_ekf_fns`` still imports
   ``make_ekf`` from this module, so the EKF pair from the previous KS
   ``kf.py`` is preserved verbatim below even though the L96 pipeline
   itself has moved to ensemble methods throughout.

Everything else -- window-aware prediction anchored at ``window_ics``,
window resets after every observation and at every window boundary, the
perturbed-observation update, the ``lax.scan``-based smoother loops --
is identical to the L96 implementation.
"""

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from typing import NamedTuple, Callable
import numpy as np

from examples.KS_prev_no_osci.utils import steps_per_window_exact


# ──────────────────────────────────────────────────────────────────────────────
# Covariance construction helpers (KS-specific)
# ──────────────────────────────────────────────────────────────────────────────

def safe_cholesky(C: jnp.ndarray, rel_jitter: float = 1e-6) -> jnp.ndarray:
    """
    Cholesky factor of a covariance, with jitter scaled to the matrix's own
    magnitude rather than a fixed absolute epsilon.

    This matters here and not in the L96 code because L96's P0/Q0 are
    diagonal and perfectly conditioned, whereas a spatially correlated
    covariance on a 256-point grid is numerically rank-deficient: a smooth
    kernel's eigenvalues decay super-exponentially, so the smallest ones
    land in float32 rounding noise and a fixed ``1e-10 * I`` is far too
    small to keep the factorisation real. Jittering by
    ``rel_jitter * mean(diag(C))`` adds a variance floor of a few parts per
    million of the marginal variance -- physically negligible, numerically
    decisive.
    """
    n = C.shape[0]
    scale = jnp.mean(jnp.diag(C))
    return jnp.linalg.cholesky(C + rel_jitter * scale * jnp.eye(n))


def periodic_gaussian_cov(
    N:        int,
    L:        float,
    sigma:    float,
    corr_len: float,
) -> jnp.ndarray:
    """
    (N, N) Gaussian-correlated covariance on a PERIODIC 1-D grid:

        C_ij = sigma^2 * exp(-d_ij^2 / (2 * corr_len^2)),
        d_ij = periodic distance between grid points i and j on [0, L).

    Why this exists (and when to use it instead of ``sigma^2 * I``)
    ---------------------------------------------------------------
    For L96 the 40 state variables are genuinely separate dynamical
    variables, so an uncorrelated ``P0 = sigma^2 I`` is a sensible
    "we're equally unsure about each of them" prior. For KS the state is
    a smooth field sampled on a fine grid: an uncorrelated prior puts
    most of its variance at the grid scale, where the fourth-order
    hyper-diffusion term annihilates it within a fraction of a window.
    The ensemble then looks well-spread at t=0 and collapses almost
    immediately, which reads as (and gets mistaken for) filter
    divergence.

    A correlation length of roughly one characteristic KS cell -- the
    system's linearly most-unstable wavelength -- puts the initial
    spread on the scales the dynamics actually amplify. Pass
    ``corr_len <= 0`` (or simply don't configure it) to fall back to the
    diagonal covariance.

    Construction (why it goes through the FFT)
    -------------------------------------------
    Because the distance is periodic, the kernel matrix is CIRCULANT: it is
    fully determined by its first row, and its eigenvalues are that row's
    DFT. Building it directly from ``exp(-d^2 / 2 lc^2)`` is positive
    semi-definite in exact arithmetic but NOT in float32 -- a smooth kernel
    on a fine grid has eigenvalues spanning many orders of magnitude, and
    the smallest ones come out slightly negative, which makes the Cholesky
    factorisations downstream return NaN.

    Taking the DFT of the first row, clipping the spectrum at zero, and
    transforming back gives the nearest circulant matrix with a
    non-negative spectrum -- the same kernel to within rounding, but PSD by
    construction. The row is then renormalised so the marginal variance is
    exactly ``sigma^2``. Use ``safe_cholesky`` (not a bare
    ``jnp.linalg.cholesky``) on the result.

    Args:
        N:        number of grid points.
        L:        physical domain length (the grid is [0, L) with spacing L/N).
        sigma:    marginal standard deviation at every grid point.
        corr_len: Gaussian correlation length, in the SAME units as ``L``.
                  Non-positive -> returns the diagonal covariance.

    Returns:
        (N, N) covariance matrix.
    """
    if corr_len is None or corr_len <= 0:
        return jnp.eye(N) * sigma ** 2

    idx = jnp.arange(N)
    # Periodic distance from grid point 0 to grid point j.
    d = jnp.minimum(idx, N - idx) * (L / N)
    row = jnp.exp(-(d ** 2) / (2.0 * corr_len ** 2))

    # Project onto the nearest non-negative circulant spectrum.
    lam = jnp.maximum(jnp.real(jnp.fft.fft(row)), 0.0)
    row = jnp.real(jnp.fft.ifft(lam))
    row = row / row[0]                              # unit marginal variance

    return (sigma ** 2) * row[(idx[None, :] - idx[:, None]) % N]


def build_cov(N: int, L: float, sigma: float, corr_len: float = 0.0) -> jnp.ndarray:
    """
    Thin dispatcher used by ``eval.py`` so P0/Q0 construction is a single
    config switch: correlated when ``corr_len > 0``, diagonal otherwise.
    """
    return periodic_gaussian_cov(N, L, sigma, corr_len)


# ──────────────────────────────────────────────────────────────────────────────
# Extended Kalman Filter (retained from the previous KS kf.py)
# ──────────────────────────────────────────────────────────────────────────────

class EKFState(NamedTuple):
    """Holds the full EKF state at a single time step."""
    x_hat:     jnp.ndarray  # (N,)   — posterior state estimate
    P:         jnp.ndarray  # (N, N) — posterior error covariance
    window_ic: jnp.ndarray  # (N,)   — IC at the start of the current window


def make_ekf(propagator_fn: Callable, N: int):
    """
    Factory that builds JIT-compiled EKF predict/update steps.

    The surrogate propagator is expected to have the signature
        propagator_fn(u: (N,), t: float) -> (N,)
    where ``t`` is the query time *within the current window*.

    Key differences from the EnKF:
    • Jacobian-based covariance propagation — the nonlinear surrogate is
      linearised at ``window_ic`` via jacfwd, giving the first-order
      covariance update F P F^T + Q.
    • Window-aware prediction — the mean and Jacobian are both evaluated
      at ``window_ic`` with query time ``t_query``, not chained from the
      previous fine-step output.
    • Deterministic update — no perturbed observations; the standard
      Kalman gain formula is used.

    Note on cost for KS: the Jacobian is (N, N) with N = 256 grid points
    (vs 40 for L96), so a single EKF predict already needs 256 forward-mode
    passes through the DeepONet and an O(N^3) covariance product. This is
    why the evaluation pipeline uses the ensemble methods below throughout;
    the EKF is kept for parity with ``models.KSUDON.make_ekf_fns``.

    Args:
        propagator_fn: callable (u: (N,), t: float) -> (N,), the surrogate
                       evaluated at a query time within the window.
                       Must be pure (no side effects).
        N: state dimension (number of grid points for KS).

    Returns:
        predict_fn, update_fn — both JIT-compiled.
    """

    @jit
    def predict(
        ekf_state: EKFState,
        Q:         jnp.ndarray,  # (N, N) process noise covariance
        t_query:   float,        # in-window time offset for this step
    ) -> EKFState:
        """
        EKF prediction step (window-aware).

            x_hat_pred = propagator_fn(window_ic, t_query)
            P_pred     = F P F^T + Q,   F = d(propagator)/d(window_ic)

        ``window_ic`` is left unchanged here; it is reset by
        ``run_ekf_smoother`` at window boundaries and after observation
        updates.
        """
        x_hat_pred = propagator_fn(ekf_state.window_ic, t_query)              # (N,)
        F = jacfwd(lambda u: propagator_fn(u, t_query))(ekf_state.window_ic)  # (N, N)
        P_pred = F @ ekf_state.P @ F.T + Q                                    # (N, N)
        return EKFState(x_hat=x_hat_pred, P=P_pred, window_ic=ekf_state.window_ic)

    @jit
    def update(
        ekf_state: EKFState,
        y_obs:     jnp.ndarray,  # (m,)
        H:         jnp.ndarray,  # (m, N)
        R:         jnp.ndarray,  # (m, m)
    ) -> tuple[EKFState, jnp.ndarray]:
        """
        EKF update step (measurement assimilation), Joseph form.

        Returns the posterior EKFState and the Kalman gain K.
        ``window_ic`` is passed through unchanged; ``run_ekf_smoother``
        is responsible for resetting it to ``x_hat_post`` after every
        update.
        """
        x_hat_pred = ekf_state.x_hat
        P_pred = ekf_state.P

        innov = y_obs - H @ x_hat_pred                          # (m,)
        S = H @ P_pred @ H.T + R                                # (m, m)
        K = jax.scipy.linalg.solve(S, (P_pred @ H.T).T, assume_a='pos').T   # (N, m)

        x_hat_post = x_hat_pred + K @ innov                     # (N,)
        I_KH = jnp.eye(N) - K @ H                               # (N, N)
        P_post = I_KH @ P_pred @ I_KH.T + K @ R @ K.T           # (N, N) Joseph form

        return EKFState(x_hat=x_hat_post, P=P_post, window_ic=ekf_state.window_ic), K

    return predict, update


def run_ekf_smoother(
    predict_fn:       Callable,
    update_fn:        Callable,
    x0_hat:           jnp.ndarray,    # (N,)
    P0:               jnp.ndarray,    # (N, N)
    observations:     jnp.ndarray,    # (T_obs, m)
    obs_step_indices: np.ndarray,     # (T_obs,) int — fine step of each obs
    H_seq:            jnp.ndarray,    # (T_obs, m, N)
    Q:                jnp.ndarray,    # (N, N) — per fine step
    R:                jnp.ndarray,    # (m, m)
    total_fine_steps: int,
    dt_fine:          float,          # fine integration step
    dt_window:        float,          # DeepONet training window length
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fine-step-centric, window-aware EKF smoother.

    Window-aware prediction
    -----------------------
    Rather than chaining f(f(u, dt_fine), dt_fine) … across fine steps,
    this smoother tracks which fine step we are at *within the current
    window* and always queries the surrogate as

        x_hat_pred = propagator_fn(window_ic, step_in_window * dt_fine)

    so that all in-window predictions exploit the full training range
    [t0, t1] and avoid Jacobian-error accumulation across fine steps.

    Window reset policy
    -------------------
    ``window_ic`` is reset to the current ``x_hat`` after every
    observation update (a mid-window observation starts a fresh
    pseudo-window from the posterior) and at every window boundary.
    Both resets also zero ``step_in_window``.

    Returns:
        x_hats:             (total_fine_steps, N) filtered state estimates.
        Ps:                 (total_fine_steps, N, N) filtered covariances.
        prior_means_at_obs: (T_obs, N) estimate *before* the update at each
                            observation step — used for the Error Reduction
                            Factor (ERF = prior RMSE / posterior RMSE).
    """
    steps_per_window = steps_per_window_exact(dt_window, dt_fine)

    # O(1) lookup: fine step index -> observation index (-1 = no obs)
    obs_at_step = np.full(total_fine_steps, -1, dtype=int)
    for obs_idx, step_idx in enumerate(np.asarray(obs_step_indices)):
        obs_at_step[step_idx] = obs_idx

    state = EKFState(x_hat=x0_hat, P=P0, window_ic=x0_hat)
    x_hats: list[jnp.ndarray] = []
    Ps: list[jnp.ndarray] = []
    prior_means_at_obs: list[jnp.ndarray] = []
    step_in_window = 0

    for fine_t in range(total_fine_steps):
        t_query = (step_in_window + 1) * dt_fine
        state = predict_fn(state, Q, t_query)

        reset_window = False
        obs_idx = obs_at_step[fine_t]
        if obs_idx >= 0:
            prior_means_at_obs.append(state.x_hat)
            state, _ = update_fn(state, observations[obs_idx], H_seq[obs_idx], R)
            reset_window = True

        x_hats.append(state.x_hat)
        Ps.append(state.P)

        step_in_window += 1
        if step_in_window >= steps_per_window:
            reset_window = True

        if reset_window:
            state = EKFState(x_hat=state.x_hat, P=state.P, window_ic=state.x_hat)
            step_in_window = 0

    return (
        jnp.stack(x_hats),
        jnp.stack(Ps),
        jnp.stack(prior_means_at_obs),  # (T_obs, N)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Kalman Filter
# ──────────────────────────────────────────────────────────────────────────────

class EnKFState(NamedTuple):
    """
    Holds the full EnKF state: an ensemble of N_ens state vectors together
    with the initial conditions used to anchor DeepONet queries for the
    current assimilation window.

    Fields
    ------
    ensemble : (N_ens, N)
        Current state estimates, including any observation updates and
        inflation applied since the last window reset. These are what gets
        assimilated when observations arrive.

    window_ics : (N_ens, N)
        Per-member initial conditions at the *start* of the current
        DeepONet window.  All predict calls within the same window query
        the surrogate as  x(t | window_ic)  for increasing t, rather than
        chaining  f(f(u, dt), dt) …  This exploits the full [t0, t1]
        training range and avoids error accumulation within a window.

        Reset to the current ensemble at every window boundary and after
        every observation update (so mid-window observations start a fresh
        pseudo-window from the assimilated state).
    """
    ensemble:   jnp.ndarray  # (N_ens, N)
    window_ics: jnp.ndarray  # (N_ens, N)


def make_enkf(propagator_fn: Callable, N: int, N_ens: int, N_dyn: int | None = None):
    """
    Factory that builds JIT-compiled EnKF predict/update steps with
    multiplicative covariance inflation.

    Args:
        propagator_fn: callable (u: (N,), t: float) -> (N,), the surrogate
                       queried at an in-window time offset.
        N:     state dimension (N grid points for KS).
        N_ens: ensemble size.
        N_dyn: number of leading components treated as DYNAMIC, i.e.
               subject to inflation. Defaults to ``N`` (the whole KS state
               is dynamic). Only set this if the state has been augmented
               with static parameters, as the L96 version is with the
               forcing F -- those trailing components are propagated but
               never inflated.

    Returns:
        predict_fn, update_fn — both JIT-compiled.
    """
    N_dyn = N if N_dyn is None else N_dyn

    @jit
    def predict(
        enkf_state: EnKFState,
        alpha:      float,        # multiplicative inflation for THIS step
        key:        jnp.ndarray,  # unused (kept for signature parity with Route B)
        t_query:    float,        # in-window time offset for this step
    ) -> EnKFState:
        """
        EnKF prediction step (window-aware) + multiplicative inflation.

        Each member is propagated by querying the surrogate at ``t_query``
        relative to that member's own ``window_ic``:

            x_pred_i = propagator_fn(window_ic_i, t_query)

        giving x(window_ic_i, t_query) directly regardless of how many fine
        steps have elapsed since the window started -- avoiding the
        cumulative error of chaining f(f(u, dt), dt) … when
        DT_FINE < DT_WINDOW.

        Anomalies about the forecast mean are then scaled by ``alpha``
        (leaving the mean untouched, so inflation changes only the spread).
        ``window_ics`` are deliberately left unchanged; they are reset by
        ``run_enkf_smoother`` at window boundaries and after observation
        updates.
        """
        # 1. Propagate every member from its window IC at t_query.
        ensemble_pred = vmap(
            lambda u: propagator_fn(u, t_query)
        )(enkf_state.window_ics)                                 # (N_ens, N)

        # 2. Ensemble mean and anomalies.
        x_mean = jnp.mean(ensemble_pred, axis=0, keepdims=True)  # (1, N)
        x_anom = ensemble_pred - x_mean                          # (N_ens, N)

        # 3. Multiplicative inflation, restricted to the dynamic block.
        inflation_mask = jnp.ones((N,)).at[:N_dyn].set(alpha)
        ensemble_inf = x_mean + x_anom * inflation_mask

        return EnKFState(
            ensemble=ensemble_inf,
            window_ics=enkf_state.window_ics,
        )

    @jit
    def update(
        enkf_state: EnKFState,
        y_obs:      jnp.ndarray,  # (m,)
        H:          jnp.ndarray,  # (m, N)
        R:          jnp.ndarray,  # (m, m)
        key:        jnp.ndarray,
    ) -> tuple[EnKFState, jnp.ndarray]:
        """
        EnKF update step — stochastic (perturbed-observation) formulation.

        The returned EnKFState carries ``window_ics`` forward unchanged;
        ``run_enkf_smoother`` resets them to the posterior ensemble after
        every update so the next predict call starts a fresh window from
        the assimilated state.

        Returns:
            Posterior EnKFState (ensemble updated, window_ics unchanged), K.
        """
        ensemble = enkf_state.ensemble   # (N_ens, N)
        m = H.shape[0]

        # Ensemble anomalies
        x_mean = jnp.mean(ensemble, axis=0)
        X_anom = ensemble - x_mean                               # (N_ens, N)

        # Predicted observations and anomalies
        y_pred = vmap(lambda x: H @ x)(ensemble)                 # (N_ens, m)
        y_mean = jnp.mean(y_pred, axis=0)
        Y_anom = y_pred - y_mean                                 # (N_ens, m)

        # Ensemble-based Kalman gain
        scale = 1.0 / (N_ens - 1)
        PHT = scale * X_anom.T @ Y_anom                            # (N, m)
        S = scale * Y_anom.T @ Y_anom + R                          # (m, m)
        K = jax.scipy.linalg.solve(S, PHT.T, assume_a='pos').T     # (N, m)

        # Perturbed observations
        L_R = jnp.linalg.cholesky(R + 1e-10 * jnp.eye(m))
        eps = jax.random.normal(key, shape=(N_ens, m)) @ L_R.T
        y_perturbed = y_obs[None, :] + eps                       # (N_ens, m)

        # Per-member update
        innovations = y_perturbed - y_pred
        ensemble_post = ensemble + innovations @ K.T             # (N_ens, N)

        return EnKFState(
            ensemble=ensemble_post,
            window_ics=enkf_state.window_ics,
        ), K

    return predict, update


def run_enkf_smoother(
    predict_fn:       Callable,
    update_fn:        Callable,
    ensemble0:        jnp.ndarray,   # (N_ens, N)
    observations:     jnp.ndarray,   # (T_obs, m)
    obs_step_indices: np.ndarray,    # (T_obs,) int — fine step of each obs
    H_seq:            jnp.ndarray,   # (T_obs, m, N)
    alpha_fine:       float,         # per-fine-step multiplicative inflation
    R:                jnp.ndarray,   # (m, m)
    key:              jnp.ndarray,
    total_fine_steps: int,
    dt_fine:          float,         # fine integration step
    dt_window:        float,         # DeepONet training window length
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fine-step-centric, window-aware EnKF smoother.

    Window-aware prediction
    -----------------------
    The DeepONet was trained on windows of length ``dt_window``.  Rather
    than chaining  f(f(u, dt_fine), dt_fine) …  across fine steps, this
    smoother tracks which fine step we are at *within the current window*
    and always queries the surrogate as

        x_pred = propagator_fn(window_ic, step_in_window * dt_fine)

    so that all in-window predictions exploit the full training range
    [t0, t1].  This avoids error accumulation that arises when the
    window is subdivided into many fine steps.

    Inflation accumulation
    ----------------------
    Because every in-window prediction restarts from ``window_ics``, the
    inflation applied at in-window step k is ``alpha_fine ** k``, i.e. the
    factor that WOULD have accumulated had the k steps been chained. Over
    a full window this compounds to the window-level factor
    ``alpha_coarse`` that ``scale_inflation_for_fine_steps`` was given.

    Window reset policy
    -------------------
    ``window_ics`` are reset to the current ensemble (1) immediately after
    an observation update, so subsequent in-window predictions start from
    the corrected state, and (2) at every window boundary, even with no
    observation. Both resets zero ``step_in_window``, so the next predict
    call queries at ``t = dt_fine``.

    Returns:
        x_means:            (total_fine_steps, N) ensemble-mean estimates.
        x_spreads:          (total_fine_steps, N) per-component ensemble std.
        prior_means_at_obs: (T_obs, N) ensemble mean *before* the update at
                            each observation step — used to compute the Error
                            Reduction Factor (ERF = prior RMSE / post RMSE).
    """
    steps_per_window = steps_per_window_exact(dt_window, dt_fine)

    # O(1) lookup: fine step index -> observation index (-1 = no obs)
    obs_at_step = jnp.full((total_fine_steps,), -1, dtype=jnp.int32)
    obs_at_step = obs_at_step.at[jnp.asarray(obs_step_indices)].set(
        jnp.arange(len(obs_step_indices))
    )

    def step(carry, fine_t):
        state, step_in_window, key = carry
        t_query = (step_in_window + 1) * dt_fine
        cumulative_alpha = alpha_fine ** (step_in_window + 1)
        key, key_pred, key_upd = jax.random.split(key, 3)

        state = predict_fn(state, cumulative_alpha, key_pred, t_query)

        obs_idx = obs_at_step[fine_t]            # precomputed, -1 = none
        has_obs = obs_idx >= 0
        safe_idx = jnp.maximum(obs_idx, 0)
        prior_mean = jnp.mean(state.ensemble, axis=0)

        state_upd = jax.lax.cond(
            has_obs,
            lambda s: update_fn(s, observations[safe_idx], H_seq[safe_idx], R, key_upd)[0],
            lambda s: s,
            state,
        )

        step_next = step_in_window + 1
        reset = has_obs | (step_next >= steps_per_window)
        new_state = EnKFState(
            ensemble=state_upd.ensemble,
            window_ics=jnp.where(reset, state_upd.ensemble, state_upd.window_ics),
        )
        return (new_state, jnp.where(reset, 0, step_next), key), \
            (jnp.mean(new_state.ensemble, 0), jnp.std(new_state.ensemble, 0), prior_mean)

    (_, _, _), (x_means, x_spreads, prior_means_all) = jax.lax.scan(
        step, (EnKFState(ensemble0, ensemble0), 0, key), jnp.arange(total_fine_steps)
    )
    prior_means_at_obs = prior_means_all[jnp.asarray(obs_step_indices)]

    return x_means, x_spreads, prior_means_at_obs


def init_ensemble(
    x0_hat: jnp.ndarray,   # (N_total,) prior mean (possibly augmented)
    P0:     jnp.ndarray,   # (N_dyn, N_dyn) prior covariance over the dynamic block
    N_ens:  int,
    key:    jnp.ndarray,
) -> jnp.ndarray:
    """
    Draw the initial ensemble from N(x0_hat, P0).

    The Cholesky factor of P0 is used, so the FULL covariance structure is
    respected -- which matters for KS, where ``P0`` is typically the
    spatially correlated ``periodic_gaussian_cov`` rather than a diagonal
    (see that function's docstring for why).

    If ``x0_hat`` is longer than ``P0`` is wide, the trailing components
    are treated as static augmented parameters and receive zero initial
    perturbation. For the un-augmented KS state the two dimensions agree
    and no padding happens.

    Returns:
        ensemble: (N_ens, N_total)
    """
    N_total = x0_hat.shape[0]
    N_dyn = P0.shape[0]

    L = safe_cholesky(P0)
    z = jax.random.normal(key, shape=(N_ens, N_dyn))
    noise = z @ L.T

    if N_dyn < N_total:
        noise = jnp.pad(noise, ((0, 0), (0, N_total - N_dyn)))

    return x0_hat[None, :] + noise


# ──────────────────────────────────────────────────────────────────────────────
# Route B: Residual-Scaled Covariance
# ──────────────────────────────────────────────────────────────────────────────
#
# `make_enkf.predict` rescales every member's anomaly by the SAME fixed
# factor every fine step. Route B instead lets each member's inflation
# reflect how badly ITS OWN surrogate trajectory is currently violating
# the governing PDE:
#
#     rho_n^i       = PDE residual of member i's surrogate trajectory,
#                     integrated over the current step
#     Q_n^i         = (alpha + beta * ||rho_n^i||^2_{L2}) * Q0
#     u_{n+1}^{f,i} = surrogate(window_ic_i, t) + xi_n^i,   xi_n^i ~ N(0, Q_n^i)
#
# alpha is a variance floor (additive inflation that never vanishes, even
# for a member whose residual is ~0). beta converts residual magnitude into
# extra spread, standing in for the stability constant of the (expensive,
# un-computed) tangent-linear propagator. Because every member shares Q0's
# spatial shape, sampling needs only ONE Cholesky factor of Q0 per step,
# rescaled per member by sqrt(alpha + beta*||rho_n^i||^2) -- no per-member
# Cholesky, no tangent-linear solve.
#
# KS note on beta's scale
# ------------------------
# For KS, rho = v_tau - (L_op v_hat + N(v_hat)) evaluated spectrally, so
# ||rho||^2 involves fourth-order spatial derivatives summed over N=256
# grid points -- a very different magnitude from L96's O(1) 40-variable
# residual, which the L96 defaults (beta ~ 250) were tuned against.
# Treat beta as needing its own calibration here; that is exactly what
# `run_route_b_inflation_sweep` is for, and `route_b_scale_mean` is
# written into the results HDF5 so the realised scale factor can be read
# off directly.


def _trapz_weights(t: jnp.ndarray) -> jnp.ndarray:
    """
    Trapezoidal quadrature weights for a 1-D (possibly non-uniform) time
    grid ``t``, shape (n_quad,), n_quad >= 2, such that
    ``sum(w * f(t)) ~= integral of f over [t[0], t[-1]]``.
    """
    dt = jnp.diff(t)
    w = jnp.zeros_like(t)
    w = w.at[0].set(dt[0] / 2)
    w = w.at[-1].set(dt[-1] / 2)
    w = w.at[1:-1].set((dt[:-1] + dt[1:]) / 2)
    return w


def residual_l2_norm_sq(
    residual_fn: Callable,
    u:           jnp.ndarray,   # (N_total,) single member's window IC
    t_quad:      jnp.ndarray,   # (n_quad,) quadrature times within the window
) -> jnp.ndarray:
    """
    Spatiotemporal L2 norm-squared of the PDE residual for ONE ensemble
    member, anchored at window IC ``u``:

        ||rho||^2_{L2(Omega x [t_quad[0], t_quad[-1]])}
            ~= sum_k  w_k * sum_j rho_j(t_k)^2

    i.e. integrated over time via the trapezoidal rule (``t_quad``) and
    summed over the spatial dimension.

    The spatial sum is left UNWEIGHTED (no dx factor), matching the L96
    implementation and the physics loss, which also averages/sums over
    the state dimension without a quadrature weight. For KS this means
    ``beta`` absorbs the constant dx -- harmless, since beta is a tuned
    coefficient, but worth knowing when comparing beta values across grid
    resolutions: halving dx (doubling N) roughly doubles ||rho||^2 for the
    same field, so a beta tuned at N=256 should be halved at N=512.

    Args:
        residual_fn: closure ``(u, t) -> (N_dyn,)``, matching the calling
            convention of ``propagator_fn`` in ``make_enkf`` (window-IC-
            anchored, in-window query time). For KS this is exactly
            ``r_net``: rho = v_tau - (L_op v_hat + N(v_hat)), the same
            residual used in the physics loss -- reused here at inference
            time, taking no gradient w.r.t. params (only the internal
            jacfwd w.r.t. t). See ``KSUDON.make_residual_fn``.
        u:      (N_total,) window IC for this member.
        t_quad: (n_quad,) quadrature times spanning the integration window.

    Returns:
        Scalar ||rho||^2 for this member.
    """
    rho = vmap(residual_fn, (None, 0))(u, t_quad)   # (n_quad, N_dyn)
    spatial_sq = jnp.sum(rho ** 2, axis=-1)          # (n_quad,)
    w = _trapz_weights(t_quad)                       # (n_quad,)
    return jnp.sum(w * spatial_sq)


def make_route_b_enkf(
    propagator_fn: Callable,
    residual_fn:   Callable,
    N:             int,
    N_ens:         int,
):
    """
    Factory for the Route B forecast step. Assimilation is untouched by
    Route B, so the returned ``update_fn`` is reused verbatim from
    ``make_enkf`` -- only the process-noise generation inside ``predict``
    changes.

    Args:
        propagator_fn: (u, t) -> (N_dyn,) surrogate forecast, e.g.
                       ``KSUDON.make_surrogate_propagator(params)``.
        residual_fn:   (u, t) -> (N_dyn,) PDE residual, same window-IC-
                       anchored convention as propagator_fn, e.g.
                       ``KSUDON.make_residual_fn(params)``.
        N:      state dimension (kept for interface parity with
                ``make_enkf``; unused internally, same as there).
        N_ens:  ensemble size.

    Returns:
        predict_route_b_fn, update_fn -- both JIT-compiled.
    """
    _, update = make_enkf(propagator_fn, N, N_ens)

    @jit
    def predict_route_b(
        enkf_state: EnKFState,
        Q0:         jnp.ndarray,   # (N_dyn, N_dyn) fixed spatial shape/structure
        alpha:      jnp.ndarray,   # scalar >= 0, variance floor
        beta:       jnp.ndarray,   # scalar >= 0, residual sensitivity
        t_quad:     jnp.ndarray,   # (n_quad,) quadrature times for THIS step
        key:        jnp.ndarray,
        t_query:    float,
    ) -> tuple[EnKFState, dict[str, jnp.ndarray]]:

        # 1. Deterministic surrogate forecast -- identical to standard predict.
        ensemble_pred = vmap(
            lambda u: propagator_fn(u, t_query)
        )(enkf_state.window_ics)

        N_total = ensemble_pred.shape[1]
        N_dyn = Q0.shape[0]

        # 2. Per-member PDE residual, evaluated on EACH member's own window
        #    IC and integrated (trapezoidally) over this step's time span.
        #    A member whose surrogate currently violates the PDE more picks
        #    up a larger ||rho_i||^2 -> more inflation: the "physics-driven,
        #    flow-dependent additive inflation" Route B is built to produce,
        #    with no tangent-linear solve required.
        resid_sq = vmap(
            lambda u: residual_l2_norm_sq(residual_fn, u, t_quad)
        )(enkf_state.window_ics)                                    # (N_ens,)

        # 3. Route B scale factor per member: s_i = alpha + beta * ||rho_i||^2
        #      alpha -> floor / additive inflation, always active -- keeps
        #               the filter from collapsing when the surrogate looks
        #               locally physics-consistent (||rho_i|| ~ 0).
        #      beta  -> how strongly a physics-violating member gets
        #               inflated; absorbs the (otherwise expensive)
        #               tangent-linear stability constant into one scalar.
        scale = alpha + beta * resid_sq                             # (N_ens,)

        # 4. Sample xi_i ~ N(0, s_i * Q0). Q0's spatial shape is SHARED
        #    across members, so factor it ONCE and rescale the per-member
        #    draw by sqrt(s_i):
        #        s_i * Q0 = (sqrt(s_i) * L0) (sqrt(s_i) * L0)^T
        #    avoiding N_ens separate Cholesky factorizations every step.
        L_Q0 = safe_cholesky(Q0)
        z = jax.random.normal(key, shape=(N_ens, N_dyn))
        noise = (z @ L_Q0.T) * jnp.sqrt(scale)[:, None]              # (N_ens, N_dyn)

        # 5. Pad for any static augmented parameters (none for plain KS).
        if N_dyn < N_total:
            noise = jnp.pad(noise, ((0, 0), (0, N_total - N_dyn)))

        new_state = EnKFState(
            ensemble=ensemble_pred + noise,
            window_ics=enkf_state.window_ics,
        )
        return new_state, {"scale": scale, "resid_sq": resid_sq}

    return predict_route_b, update


def run_enkf_smoother_route_b(
    predict_route_b_fn: Callable,
    update_fn:          Callable,
    ensemble0:          jnp.ndarray,   # (N_ens, N)
    observations:       jnp.ndarray,   # (T_obs, m)
    obs_step_indices:   np.ndarray,    # (T_obs,) int
    H_seq:              jnp.ndarray,   # (T_obs, m, N)
    Q0:                 jnp.ndarray,   # (N_dyn, N_dyn) fixed shape covariance
    alpha:              float,
    beta:               float,
    R:                  jnp.ndarray,   # (m, m)
    key:                jnp.ndarray,
    total_fine_steps:   int,
    dt_fine:            float,
    dt_window:          float,
    n_quad:             int = 3,       # quadrature points per fine step
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Route B counterpart of ``run_enkf_smoother``: identical fine-step /
    window-reset / observation bookkeeping, but calls
    ``predict_route_b_fn(state, Q0, alpha, beta, t_quad, key, t_query)``
    instead of the fixed-inflation ``predict_fn``.

    Residual integration window
    ----------------------------
    The residual for each fine step is integrated (trapezoidally, over
    ``n_quad`` points) across
    ``[step_in_window * dt_fine, (step_in_window + 1) * dt_fine]`` -- i.e.
    the time span THIS predict call actually advances. This mirrors Q0
    being the per-fine-step base covariance, exactly as ``Q_fine`` already
    is for the standard filter (see ``scale_Q_for_fine_steps``). To
    integrate the residual over the FULL DeepONet window instead, replace
    the ``t_a, t_b`` lines below with ``t_a, t_b = 0.0, dt_window``.

    Returns:
        x_means, x_spreads, prior_means_at_obs: identical in shape/meaning
            to ``run_enkf_smoother``.
        Q_scale_history: (total_fine_steps, N_ens) per-member Route B scale
            factor s_i at every fine step -- useful for tuning alpha/beta
            and for sanity-checking that inflation tracks known model error
            (for KS, expect it to spike where the field develops sharp
            gradients and in long observation gaps).
    """
    steps_per_window = steps_per_window_exact(dt_window, dt_fine)

    obs_at_step = jnp.full((total_fine_steps,), -1, dtype=jnp.int32)
    obs_at_step = obs_at_step.at[jnp.asarray(obs_step_indices)].set(
        jnp.arange(len(obs_step_indices))
    )

    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)

    def step(carry, fine_t):
        state, step_in_window, key = carry
        t_query = (step_in_window + 1) * dt_fine

        t_a = step_in_window * dt_fine
        t_b = t_query
        t_quad = jnp.linspace(t_a, t_b, n_quad)

        key, key_pred, key_upd = jax.random.split(key, 3)

        # Predict
        state, diag = predict_route_b_fn(state, Q0, alpha, beta, t_quad, key_pred, t_query)

        obs_idx = obs_at_step[fine_t]
        has_obs = obs_idx >= 0
        safe_idx = jnp.maximum(obs_idx, 0)
        prior_mean = jnp.mean(state.ensemble, axis=0)

        # Update
        state_upd = jax.lax.cond(
            has_obs,
            lambda s: update_fn(s, observations[safe_idx], H_seq[safe_idx], R, key_upd)[0],
            lambda s: s,
            state,
        )

        step_next = step_in_window + 1
        reset = has_obs | (step_next >= steps_per_window)

        new_state = EnKFState(
            ensemble=state_upd.ensemble,
            window_ics=jnp.where(reset, state_upd.ensemble, state_upd.window_ics),
        )

        return (new_state, jnp.where(reset, 0, step_next), key), \
            (jnp.mean(new_state.ensemble, 0), jnp.std(new_state.ensemble, 0),
             prior_mean, diag["scale"])

    (_, _, _), (x_means, x_spreads, prior_means_all, Q_scale_history) = jax.lax.scan(
        step, (EnKFState(ensemble0, ensemble0), 0, key), jnp.arange(total_fine_steps)
    )
    prior_means_at_obs = prior_means_all[jnp.asarray(obs_step_indices)]

    return x_means, x_spreads, prior_means_at_obs, Q_scale_history


# ──────────────────────────────────────────────────────────────────────────────
# RTPP: Relaxation-to-Prior Perturbations
# ──────────────────────────────────────────────────────────────────────────────

def make_rtpp_enkf(propagator_fn: Callable, N: int, N_ens: int, N_dyn: int | None = None):
    """
    Factory that builds JIT-compiled EnKF predict/update steps with
    Relaxation-to-Prior Perturbations (RTPP) inflation:

        x'_post  <-  (1 - alpha_rtpp) * x'_post + alpha_rtpp * x'_prior

    applied to the posterior ANOMALIES only (the posterior mean is left
    exactly as the Kalman update produced it), so RTPP restores part of the
    spread the update removed without moving the analysis.

    The predict step is shared with ``make_enkf`` -- pass ``alpha = 1.0``
    through the smoother to isolate RTPP as the only active inflation
    mechanism, or a value > 1 to combine it with multiplicative inflation.
    """
    N_dyn = N if N_dyn is None else N_dyn
    predict, _ = make_enkf(propagator_fn, N, N_ens, N_dyn)

    @jit
    def update_rtpp(
        enkf_state: EnKFState,
        y_obs:      jnp.ndarray,
        H:          jnp.ndarray,
        R:          jnp.ndarray,
        alpha_rtpp: float,        # RTPP relaxation parameter in [0, 1]
        key:        jnp.ndarray,
    ) -> tuple[EnKFState, jnp.ndarray]:

        prior_ensemble = enkf_state.ensemble
        m = H.shape[0]

        # 1. Prior anomalies
        x_mean_prior = jnp.mean(prior_ensemble, axis=0)
        X_anom_prior = prior_ensemble - x_mean_prior

        # 2. Standard stochastic assimilation
        y_pred = vmap(lambda x: H @ x)(prior_ensemble)
        y_mean = jnp.mean(y_pred, axis=0)
        Y_anom = y_pred - y_mean

        scale = 1.0 / (N_ens - 1)
        PHT = scale * X_anom_prior.T @ Y_anom
        S = scale * Y_anom.T @ Y_anom + R
        K = jax.scipy.linalg.solve(S, PHT.T, assume_a='pos').T

        L_R = jnp.linalg.cholesky(R + 1e-10 * jnp.eye(m))
        eps = jax.random.normal(key, shape=(N_ens, m)) @ L_R.T
        y_perturbed = y_obs[None, :] + eps

        innovations = y_perturbed - y_pred
        ensemble_post = prior_ensemble + innovations @ K.T

        # 3. Posterior anomalies
        x_mean_post = jnp.mean(ensemble_post, axis=0)
        X_anom_post = ensemble_post - x_mean_post

        # 4. RTPP relaxation, restricted to the dynamic block
        rtpp_mask = jnp.zeros((N,)).at[:N_dyn].set(alpha_rtpp)

        X_anom_rtpp = (1.0 - rtpp_mask) * X_anom_post + rtpp_mask * X_anom_prior
        ensemble_rtpp = x_mean_post + X_anom_rtpp

        return EnKFState(
            ensemble=ensemble_rtpp,
            window_ics=enkf_state.window_ics,
        ), K

    return predict, update_rtpp


def run_enkf_smoother_rtpp(
    predict_fn:       Callable,
    update_fn:        Callable,
    ensemble0:        jnp.ndarray,
    observations:     jnp.ndarray,
    obs_step_indices: np.ndarray,
    H_seq:            jnp.ndarray,
    alpha_fine:       float,         # multiplicative inflation (predict step)
    alpha_rtpp:       float,         # RTPP relaxation factor (update step)
    R:                jnp.ndarray,
    key:              jnp.ndarray,
    total_fine_steps: int,
    dt_fine:          float,
    dt_window:        float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Smoother variant for RTPP: identical bookkeeping to
    ``run_enkf_smoother``, but passes ``alpha_rtpp`` through to the
    RTPP-specific update step.
    """
    steps_per_window = steps_per_window_exact(dt_window, dt_fine)
    obs_at_step = jnp.full((total_fine_steps,), -1, dtype=jnp.int32)
    obs_at_step = obs_at_step.at[jnp.asarray(obs_step_indices)].set(
        jnp.arange(len(obs_step_indices))
    )

    def step(carry, fine_t):
        state, step_in_window, key = carry
        t_query = (step_in_window + 1) * dt_fine
        cumulative_alpha = alpha_fine ** (step_in_window + 1)
        key, key_pred, key_upd = jax.random.split(key, 3)

        state = predict_fn(state, cumulative_alpha, key_pred, t_query)

        obs_idx = obs_at_step[fine_t]
        has_obs = obs_idx >= 0
        safe_idx = jnp.maximum(obs_idx, 0)
        prior_mean = jnp.mean(state.ensemble, axis=0)

        state_upd = jax.lax.cond(
            has_obs,
            lambda s: update_fn(
                s, observations[safe_idx], H_seq[safe_idx], R, alpha_rtpp, key_upd
            )[0],
            lambda s: s,
            state,
        )

        step_next = step_in_window + 1
        reset = has_obs | (step_next >= steps_per_window)
        new_state = EnKFState(
            ensemble=state_upd.ensemble,
            window_ics=jnp.where(reset, state_upd.ensemble, state_upd.window_ics),
        )
        return (new_state, jnp.where(reset, 0, step_next), key), \
            (jnp.mean(new_state.ensemble, 0), jnp.std(new_state.ensemble, 0), prior_mean)

    (_, _, _), (x_means, x_spreads, prior_means_all) = jax.lax.scan(
        step, (EnKFState(ensemble0, ensemble0), 0, key), jnp.arange(total_fine_steps)
    )
    prior_means_at_obs = prior_means_all[jnp.asarray(obs_step_indices)]

    return x_means, x_spreads, prior_means_at_obs