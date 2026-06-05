import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz96_deriv(t, x, F):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def compute_and_plot_l96_decorrelation(
        J=40, 
        F=6.0, 
        dt=0.01, 
        T_total=1000.0, 
        T_transient=300.0, 
        save_path="examples/l96_n40_f6_ics/decorrelation_t/plot.pdf"
        ):
    t_transient = np.arange(0, T_transient, dt)
    t_eval = np.arange(0, T_total, dt)
    
    x0 = np.full(J, F, dtype=float)
    x0[J // 2] += F * 0.001 
    
    sol_trans = solve_ivp(fun=lorenz96_deriv, t_span=(0, T_transient), y0=x0, args=(F,), t_eval=t_transient, method='RK45')
    
    x0_main = sol_trans.y[:, -1]
    sol_main = solve_ivp(fun=lorenz96_deriv, t_span=(0, T_total), y0=x0_main, args=(F,), t_eval=t_eval, method='RK45')
    
    X = sol_main.y
    num_steps = X.shape[1]
    lags = np.arange(num_steps) * dt
    
    acf_sum = np.zeros(num_steps)
    for j in range(J):
        xj = X[j, :]
        xj_centered = xj - np.mean(xj)
        fft_xj = np.fft.fft(xj_centered, n=2 * num_steps)
        power_spectrum = np.abs(fft_xj)**2
        acf_j_full = np.fft.ifft(power_spectrum).real
        acf_j = acf_j_full[:num_steps]
        acf_j /= acf_j[0]
        acf_sum += acf_j
        
    acf_avg = acf_sum / J
    sign_changes = np.where(np.diff(np.sign(acf_avg)))[0]
    
    tau_d = np.nan
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        t1, t2 = lags[idx], lags[idx + 1]
        y1, y2 = acf_avg[idx], acf_avg[idx + 1]
        tau_d = t1 - y1 * (t2 - t1) / (y2 - y1)

    plt.figure(figsize=(10, 6))
    plt.plot(lags, acf_avg, label='Ensemble Averaged ACF', color='#1f77b4', linewidth=2)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    if not np.isnan(tau_d):
        plt.axvline(tau_d, color='#d62728', linestyle='--', label=f'Decorrelation Time $\\approx {tau_d:.3f}$')
        plt.plot(tau_d, 0, marker='o', markersize=8, color='#d62728')
        plt.xlim(0, max(5.0, tau_d * 4))
    else:
        plt.xlim(0, T_total / 4)

    plt.title(f"Autocorrelation & Decorrelation Time for Lorenz '96 ($J={J}, F={F}$)")
    plt.xlabel("Lag $\\tau$")
    plt.ylabel("Normalized Autocorrelation $\\bar{R}(\\tau)$")
    plt.ylim(-0.5, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    compute_and_plot_l96_decorrelation(J=40, F=6.0)