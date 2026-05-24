import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing

# ---------------- Core utilities ----------------
def expand_atoms_to_eigs(lambdas, weights, d):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    counts = np.maximum(1, np.round(weights * d).astype(int))
    diff = d - counts.sum()
    if diff != 0:
        order = np.argsort(-weights)
        for k in order:
            if diff == 0: break
            counts[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
    eigs = []
    for lam, c in zip(lambdas, counts):
        eigs.extend([lam] * int(c))
    eigs = np.array(eigs, dtype=float)
    if eigs.size > d: eigs = eigs[:d]
    if eigs.size < d: eigs = np.pad(eigs, (0, d - eigs.size), constant_values=lambdas[-1])
    return eigs

def simulate_sample_cov_eigs(pop_eigs, n, seed=0):
    rng = np.random.default_rng(seed)
    d = len(pop_eigs)
    sqrt_Sigma = np.sqrt(pop_eigs)
    Z = rng.standard_normal(size=(d, n)) / np.sqrt(n)
    X = sqrt_Sigma[:, None] * Z
    S = X @ X.T
    evals = np.linalg.eigvalsh(S)
    return np.sort(evals)

def stieltjes_empirical(tilde_z, evals):
    evals = np.asarray(evals, dtype=float)
    tz = np.asarray(tilde_z, dtype=complex)
    denom = evals[None, :] - tz[..., None]
    return (1.0 / len(evals)) * np.sum(1.0 / denom, axis=-1)

def varphi_empirical(tilde_z, evals, c_n):
    s = stieltjes_empirical(tilde_z, evals)
    denom = 1.0 - c_n * tilde_z * s - c_n
    return tilde_z / denom, denom

def stieltjes_H_atoms(z, atoms, weights):
    z = np.asarray(z, dtype=complex)
    s = np.zeros_like(z, dtype=complex)
    for lam, w in zip(atoms, weights):
        s += w / (lam - z)
    return s

def Phi_from_s(z, c, sH):
    return (1.0 - c * z * sH - c) * z

# ---------------- Main plot ----------------
def plot_varphi_empirical_grid(pop_eigs=None,
                               atoms=None, weights=None, d=None,
                               n=2000, seed=0,
                               n_vert=7, n_horz=5,
                               eps_min=0.05,
                               x_pad_ratio=0.08,
                               y_max=None,
                               base_bins=500,
                               lw_grid=2.0,
                               style_horz='--',
                               blowup_tol=1e-7,
                               max_abs=1e6,
                               left_bins=900,
                               morph_close_iters=2,
                               morph_dilate_iters=2,
                               contour_res=600,
                               left_res=600,
                               # histogram controls
                               show_hist=True,
                               hist_bins=60,
                               hist_height_frac=0.18,
                               hist_color='lightgray',
                               # output controls
                               save_path=None,
                               show=True):
    """
    Finite-sample visualization with right-panel histogram and left/right blue fills.
    Also draws green boundaries on the left.
    """

    # --- Population setup ---
    if pop_eigs is None:
        assert (atoms is not None) and (weights is not None) and (d is not None)
        weights = np.asarray(weights, dtype=float); weights /= weights.sum()
        atoms = np.asarray(atoms, dtype=float)
        pop_eigs = expand_atoms_to_eigs(atoms, weights, d)
        H_atoms, H_weights = atoms, weights
    else:
        pop_eigs = np.asarray(pop_eigs, dtype=float)
        d = len(pop_eigs)
        uniq, cnt = np.unique(pop_eigs, return_counts=True)
        H_atoms, H_weights = uniq, cnt / cnt.sum()

    d = int(d); n = int(n)
    c_n = d / float(n)

    # --- Empirical eigenvalues (for ν̂_n) ---
    evals = simulate_sample_cov_eigs(pop_eigs, n=n, seed=seed)

    # --- Right (tilde-z) window ---
    xmin, xmax = float(evals.min()), float(evals.max())
    #xmin, xmax = 1,4
    x_span = xmax - xmin if xmax > xmin else max(1.0, abs(xmax))
    pad = x_pad_ratio * x_span
    x_lo, x_hi = xmin - pad, xmax + pad
    y_hi = float(y_max) if (y_max is not None) else max(1.0, 0.6 * x_span + 0.5)
    y_lo_plot = 0.0
    y_lo_fill = float(eps_min)

    # --- Grid in tilde-z (right) ---
    res = int(base_bins)
    xs = np.linspace(x_lo, x_hi, res)
    ys = np.linspace(y_lo_fill, y_hi, res)
    Xr, Yr = np.meshgrid(xs, ys)
    T = Xr + 1j * Yr

    # --- Map region for left fill ---
    s_emp = stieltjes_empirical(T, evals)
    denom = 1.0 - c_n * T * s_emp - c_n
    Z_img = T / denom
    ok = np.isfinite(Z_img) & np.isfinite(denom)
    ok &= (np.abs(denom) > blowup_tol)
    ok &= (np.abs(Z_img) < max_abs)
    ok &= (np.imag(Z_img) >= 0.0)
    mask_valid = ok.reshape(T.shape)
    pts = Z_img[mask_valid]
    have_pts = pts.size > 0

    # --- Figure setup ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax in (axL, axR): ax.set_facecolor("white")
    axL.set_title(r"$H$-space")
    axR.set_title(r"$\nu$-space")

    # --- Right-panel histogram (background) ---
    if show_hist and evals.size > 0:
        counts, edges = np.histogram(evals, bins=hist_bins, range=(x_lo, x_hi), density=False)
        if counts.max() > 0:
            heights = (counts / counts.max()) * (hist_height_frac * y_hi)
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = (edges[1] - edges[0]) * 0.9
            axR.bar(centers, heights, width=width,
                    bottom=0.0, color=hist_color, edgecolor='none',
                    alpha=0.9, zorder=0)

    # --- RIGHT: blue fill in the grid band ---
    axR.imshow(np.ones_like(mask_valid, dtype=float),
               extent=[x_lo, x_hi, y_lo_fill, y_hi],
               origin='lower', aspect='auto',
               cmap='Blues', alpha=0.4, vmin=0, vmax=1,
               interpolation='nearest', zorder=1)

    # --- LEFT: blue fill via raster mask (no mesh lines) ---
    if have_pts:
        re, im = np.real(pts), np.imag(pts)
        rmin, rmax = re.min(), re.max()
        imin, imax = im.min(), im.max()
        pad_r = 0.05 * (rmax - rmin + 1e-12)
        pad_i = 0.05 * (imax - imin + 1e-12)
        left_xlim = (rmin - pad_r, rmax + pad_r)
        left_ylim = (max(0.0, imin - pad_i), imax + pad_i)
        Hb, r_edges, i_edges = np.histogram2d(re, im, bins=left_bins, range=[left_xlim, left_ylim])
        mask_left = Hb.T > 0
        mask_left = binary_closing(mask_left, iterations=morph_close_iters)
        mask_left = binary_fill_holes(mask_left)
        if morph_dilate_iters > 0:
            mask_left = binary_dilation(mask_left, iterations=morph_dilate_iters)
        extent_left = [r_edges[0], r_edges[-1], i_edges[0], i_edges[-1]]
        axL.imshow(mask_left.astype(float),
                   extent=extent_left, origin='lower', aspect='auto',
                   cmap='Blues', alpha=0.4, vmin=0, vmax=1,
                   interpolation='nearest', zorder=1)
    else:
        extent_left = [x_lo, x_hi, y_lo_fill, y_hi]

    # --- Black grids (both sides) ---
    grid_color = "black"
    js = np.linspace(0, res - 1, n_vert, dtype=int)
    is_ = np.linspace(0, res - 1, n_horz, dtype=int)

    for j in js:
        t_line = T[:, j]
        axR.plot(np.real(t_line), np.imag(t_line), lw=lw_grid, color=grid_color, zorder=2)
        z_line, dnm = varphi_empirical(t_line, evals, c_n)
        good = (np.isfinite(z_line) & np.isfinite(dnm) &
                (np.abs(dnm) > blowup_tol) & (np.imag(z_line) >= 0) &
                (np.abs(z_line) < max_abs))
        axL.plot(np.real(z_line[good]), np.imag(z_line[good]),
                 lw=lw_grid, color=grid_color, zorder=2)
    for i in is_:
        t_line = T[i, :]
        axR.plot(np.real(t_line), np.imag(t_line), lw=lw_grid, color=grid_color, linestyle=style_horz, zorder=2)
        z_line, dnm = varphi_empirical(t_line, evals, c_n)
        good = (np.isfinite(z_line) & np.isfinite(dnm) &
                (np.abs(dnm) > blowup_tol) & (np.imag(z_line) >= 0) &
                (np.abs(z_line) < max_abs))
        axL.plot(np.real(z_line[good]), np.imag(z_line[good]),
                 lw=lw_grid, color=grid_color, linestyle=style_horz, zorder=2)

    # --- Green boundaries on LEFT (solid & dashed) ---
    zx = np.linspace(extent_left[0], extent_left[1], contour_res)
    zy_low = max(1e-4, extent_left[2])
    zy = np.linspace(zy_low, extent_left[3], left_res)
    ZX, ZY = np.meshgrid(zx, zy)
    Zz = ZX + 1j * ZY

    # Solid: ∂D_{H_n,c_n}(0,∞)
    s_Hn = np.mean(1.0 / (pop_eigs[None, ...] - Zz[..., None]), axis=-1)
    ImPhi_Hn = np.imag((1.0 - c_n * Zz * s_Hn - c_n) * Zz)
    axL.contour(ZX, ZY, ImPhi_Hn, levels=[0.0], colors='green', linewidths=2.2, zorder=3)

    # Dashed: ∂D_{H,c}(0,1)
    s_H = stieltjes_H_atoms(Zz, H_atoms, H_weights)
    Phi_H = (1.0 - c_n * Zz * s_H - c_n) * Zz
    ImPhi_H = np.imag(Phi_H)
    axL.contour(ZX, ZY, ImPhi_H, levels=[0.0],
                colors='green', linewidths=2.0, linestyles='--', zorder=3)
    tiny = 1e-12
    num = np.abs(c_n * Zz * np.imag(Zz * s_H))
    den = np.maximum(ImPhi_H, tiny)
    ratio = num / den - 1.0
    ratio_masked = np.ma.masked_where(ImPhi_H <= 0.0, ratio)
    try:
        axL.contour(ZX, ZY, ratio_masked, levels=[0.0],
                    colors='green', linewidths=2.0, linestyles='--', zorder=3)
    except Exception:
        pass

    # --- labels/limits ---
    axR.set_xlabel(r"Re($\tilde{z}$)"); axR.set_ylabel(r"Im($\tilde{z}$)")
    axR.set_xlim(x_lo, x_hi); axR.set_ylim(y_lo_plot, y_hi)
    axL.set_xlabel(r"Re($z$)"); axL.set_ylabel(r"Im($z$)")

    # --- output ---
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    if show:
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"plt.show() failed: {e}")

# ---------------- Run an example when executed as a script ----------------
#etaMin=0.1
etaMin=0.02

NN=30
if __name__ == "__main__":
    d = 200
    PopEV = [1]*(d//2)+[2.5+i/d for i in range(d-d//2)]
    plot_varphi_empirical_grid(
        atoms=PopEV, weights=[0.5]+[1/2/NN]*NN,
        d=d, n=10*d, seed=1,
        n_vert=8, n_horz=6,
        eps_min=etaMin, y_max=1.5,
        left_res=1000,
        # histogram:
        show_hist=True, hist_bins=40, hist_height_frac=0.5, hist_color='lightgray',
        # output:
        save_path="finite_sample_plot.png", show=True, pop_eigs=PopEV
    )

if 0:
    plot_varphi_empirical_grid(
        atoms=[0.5, 1.0], weights=[0.5, 0.5],
        d=200, n=4000, seed=1,
        n_vert=8, n_horz=6,
        eps_min=etaMin, y_max=0.5,
        left_res=3000,
        # histogram:
        show_hist=True, hist_bins=70, hist_height_frac=0.18, hist_color='lightgray',
        # output:
        save_path="finite_sample_plot.png", show=True
    )
