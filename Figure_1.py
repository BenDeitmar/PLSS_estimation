import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_dilation

# ================== Core transforms ==================
def stieltjes_H(z, lambdas, weights):
    z = np.asarray(z, dtype=complex)
    s = np.zeros_like(z, dtype=complex)
    for lam, w in zip(lambdas, weights):
        s += w / (lam - z)
    return s

def Phi_H_c(z, c, lambdas, weights):
    sH = stieltjes_H(z, lambdas, weights)
    return (1 - c*z*sH - c) * z

# ================== Domain utilities ==================
def touches_border(mask, ignore_bottom=True):
    if mask.size == 0:
        return False
    top = mask[-1, :].any()
    left = mask[:, 0].any()
    right = mask[:, -1].any()
    bottom = (False if ignore_bottom else mask[0, :].any())
    return top or left or right or bottom

def find_window_auto(lambdas, weights, c,
                     base_half_width=3.0, base_ymax=3.0,
                     res=500, max_iter=8, growth_factor=1.8,
                     y_eps=1e-3):
    """Expand until complement is interior (ignoring bottom border)."""
    half_w = float(base_half_width)
    ymax = float(base_ymax)
    last = None
    for _ in range(max_iter):
        x = np.linspace(-half_w, half_w, res)
        y = np.linspace(y_eps, ymax, res)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y
        Phi = Phi_H_c(Z, c, lambdas, weights)
        comp = (np.imag(Phi) <= 0)
        last = ((-half_w, half_w), (y_eps, ymax), X, Y, Phi, comp)
        if not np.any(comp):
            half_w *= growth_factor; ymax *= growth_factor; continue
        if touches_border(comp, ignore_bottom=True):
            half_w *= growth_factor; ymax *= growth_factor; continue
        return (-half_w, half_w), (y_eps, ymax), X, Y, Phi, comp
    return last

def tighten_window_to_complement(xlim, ylim, comp, X, Y, pad_ratio=0.2):
    ys, xs = np.where(comp)
    if ys.size == 0:
        return xlim, ylim
    x_vals = X[0, :]; y_vals = Y[:, 0]
    xmin, xmax = x_vals[xs].min(), x_vals[xs].max()
    ymin, ymax = y_vals[ys].min(), y_vals[ys].max()
    pad_x = pad_ratio * max(1e-8, xmax - xmin)
    pad_y = pad_ratio * max(1e-8, ymax - ymin)
    xmin -= pad_x; xmax += pad_x
    ymin = max(ylim[0], ymin - pad_y)
    ymax += pad_y
    #return (xmin, xmax), (ylim[0], ymax)
    #return (xmin, xmax), (ylim[0], 0.5)
    return (0, 5), (ylim[0], 2)

# ================== Grid drawing helpers ==================
def _plot_boundary(ax, X, Y, Phi):
    # return contour set (Im Phi = 0) for later extraction
    CS = ax.contour(X, Y, np.imag(Phi), levels=[0.0], colors='black', linewidths=1.0)
    return CS

def _find_segments_on_mask(xs):
    segs = []; in_seg = False; start = 0
    for i, val in enumerate(xs):
        if val and not in_seg:
            in_seg = True; start = i
        elif not val and in_seg:
            in_seg = False; segs.append((start, i-1))
    if in_seg: segs.append((start, len(xs)-1))
    return segs

def _draw_reference_grid_left_only(ax_left, X, Y, D_mask, n_vert=7, n_horz=5, lw=1.6):
    xs_idx = np.linspace(0, X.shape[1]-1, n_vert, dtype=int)
    for j in xs_idx:
        col_mask = D_mask[:, j]
        for (i0, i1) in _find_segments_on_mask(col_mask):
            z_seg = X[i0:i1+1, j] + 1j*Y[i0:i1+1, j]
            ax_left.plot(np.real(z_seg), np.imag(z_seg), lw=lw, color='k')
    ys_idx = np.linspace(0, X.shape[0]-1, n_horz, dtype=int)
    for i in ys_idx:
        row_mask = D_mask[i, :]
        for (j0, j1) in _find_segments_on_mask(row_mask):
            z_seg = X[i, j0:j1+1] + 1j*Y[i, j0:j1+1]
            ax_left.plot(np.real(z_seg), np.imag(z_seg), lw=lw, color='k')

def _draw_mapped_grid_right_only(ax_right, X, Y, D_mask, Phi, n_vert=7, n_horz=5, lw=1.6):
    xs_idx = np.linspace(0, X.shape[1]-1, n_vert, dtype=int)
    for j in xs_idx:
        col_mask = D_mask[:, j]
        for (i0, i1) in _find_segments_on_mask(col_mask):
            phi_seg = Phi[i0:i1+1, j]
            ax_right.plot(np.real(phi_seg), np.imag(phi_seg), lw=lw, color='k')
    ys_idx = np.linspace(0, X.shape[0]-1, n_horz, dtype=int)
    for i in ys_idx:
        row_mask = D_mask[i, :]
        for (j0, j1) in _find_segments_on_mask(row_mask):
            phi_seg = Phi[i, j0:j1+1]
            ax_right.plot(np.real(phi_seg), np.imag(phi_seg), lw=lw, color='k')

# ================== Densify a contour polyline ==================
def _densify_polyline(seg_xy, factor=12):
    if seg_xy.shape[0] < 2 or factor <= 1:
        return seg_xy
    xs, ys = seg_xy[:,0], seg_xy[:,1]
    xs_new = [xs[0]]; ys_new = [ys[0]]
    for k in range(len(xs)-1):
        x0, y0 = xs[k], ys[k]
        x1, y1 = xs[k+1], ys[k+1]
        for t in range(1, factor+1):
            a = t/(factor+0.0)
            xs_new.append((1-a)*x0 + a*x1)
            ys_new.append((1-a)*y0 + a*y1)
    return np.column_stack([np.array(xs_new), np.array(ys_new)])

# ================== Main plotting ==================
def plot_D_and_image(lambdas, weights, c,
                     base_half_width=3.0, base_ymax=3.0,
                     res=900, y_eps=1e-3,
                     n_vert_lines=7, n_horz_lines=5):
    # --- Compute domain ---
    xlim, ylim, X, Y, Phi, comp = find_window_auto(lambdas, weights, c)
    xlim_t, ylim_t = tighten_window_to_complement(xlim, ylim, comp, X, Y)

    x = np.linspace(xlim_t[0], xlim_t[1], res)
    y = np.linspace(ylim_t[0], ylim_t[1], res)
    X2, Y2 = np.meshgrid(x, y)
    Z2 = X2 + 1j*Y2
    Phi2 = Phi_H_c(Z2, c, lambdas, weights)
    D_mask = (np.imag(Phi2) > 0)

    # --- Blue background for ν-plane (right) ---
    Phi_flat = Phi2[D_mask]
    Phi_re, Phi_im = np.real(Phi_flat), np.imag(Phi_flat)
    Phi_re, Phi_im = Phi_re[Phi_im > 0], Phi_im[Phi_im > 0]
    re_min, re_max = Phi_re.min(), Phi_re.max()
    im_min, im_max = Phi_im.min(), Phi_im.max()
    pad_r = 0.05 * (re_max - re_min + 1e-12)
    pad_i = 0.05 * (im_max - im_min + 1e-12)
    img_xlim = (re_min - pad_r, re_max + pad_r)
    img_ylim = (max(0, im_min - pad_i), im_max + pad_i)
    H2d, r_edges, i_edges = np.histogram2d(
        Phi_re, Phi_im, bins=800, range=[img_xlim, img_ylim]
    )
    img_mask = (H2d.T > 0)
    img_mask = binary_dilation(binary_fill_holes(img_mask), iterations=2)
    extent_right = [r_edges[0], r_edges[-1], i_edges[0], i_edges[-1]]

    # --- Figure setup ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 4.5))
    alpha_val = 0.4

    # --- Left: z-plane ---
    extent_left = [xlim_t[0], xlim_t[1], ylim_t[0], ylim_t[1]]
    axL.imshow(D_mask.astype(float), extent=extent_left, origin='lower',
               aspect='auto', cmap='Blues', alpha=alpha_val)
    CS_left = _plot_boundary(axL, X2, Y2, Phi2)
    _draw_reference_grid_left_only(axL, X2, Y2, D_mask,
                                   n_vert=n_vert_lines, n_horz=n_horz_lines, lw=1.6)
    if lambdas:
        axL.scatter(lambdas, [0.0]*len(lambdas),
                    s=60, color='orange', zorder=5,
                    label=r'$\mathrm{supp}(H)$')
        axL.legend(loc='upper right')
    axL.set_title(r"$H$-space")
    axL.set_xlabel("Re(z)"); axL.set_ylabel("Im(z)")
    axL.set_facecolor("white")

    # --- Right: ν-plane ---
    axR.imshow(img_mask.astype(float), extent=extent_right, origin='lower',
               aspect='auto', cmap='Blues', alpha=alpha_val)
    _draw_mapped_grid_right_only(axR, X2, Y2, D_mask, Phi2,
                                 n_vert=n_vert_lines, n_horz=n_horz_lines, lw=1.6)
    axR.set_title(r"$\nu$-space")
    axR.set_xlabel(r"Re($\tilde z$)"); axR.set_ylabel(r"Im($\tilde z$)")
    axR.set_facecolor("white")

    # --- Highlight the upper boundary (magenta) and its image ---
    support_color = 'green'
    intervals_all = []
    if CS_left.allsegs and len(CS_left.allsegs[0]) > 0:
        for seg in CS_left.allsegs[0]:
            seg = seg[seg[:,1] > 0]
            if seg.shape[0] < 2:
                continue
            seg_dense = _densify_polyline(seg, factor=12)
            z_b = seg_dense[:,0] + 1j*seg_dense[:,1]

            # Draw boundary on left
            axL.plot(np.real(z_b), np.imag(z_b),
                     color=support_color, lw=2.5, zorder=6,
                     label=r'$\partial\mathbb{D}_{H,c}(\infty) \cap \mathbb{C}^+$')

            # Map boundary -> ν-plane support
            phi_b = Phi_H_c(z_b, c, lambdas, weights)
            phi_b = phi_b[np.isfinite(phi_b)]
            if phi_b.size == 0:
                continue
            xr = np.real(phi_b[np.abs(np.imag(phi_b)) < 1e-6])
            if xr.size == 0:
                continue
            xr.sort()
            axR.plot(xr, np.zeros_like(xr),
                     color=support_color, lw=3.0, solid_capstyle='round', zorder=6,
                     label=r'$\mathrm{supp}(\nu)$')

    # --- Deduplicate legends ---
    for ax in (axL, axR):
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc='upper right')

    plt.tight_layout()
    #sig=1
    #x = sig/2*(c+2-np.sqrt(c*(c+8)))
    #axL.scatter([x],[0],color='green')
    #print(c*x*sig/np.abs((x-sig)**2-c*sig**2))
    upper = ax.get_ylim()[1]
    #upper = ax.get_ylim()[1]
    axR.set_ylim(-0.05*upper, upper)
    #sig=1
    #axL.scatter([abs((x-sig)**2-c*sig**2)/c/sig],[0],color='green')
    plt.show()

# ---------------- Examples ----------------
#plot_D_and_image(lambdas=[0.5, 1.0], weights=[0.5, 0.5], c=0.05)
NN=30
plot_D_and_image(lambdas=[1]+[2.5+i/2/NN for i in range(NN)], weights=[0.5]+[1/NN]*NN, c=0.1)
#plot_D_and_image(lambdas=[0.5, 1.0], weights=[0, 1], c=0.05)
# plot_D_and_image(lambdas=[1.0, 2.0], weights=[0.5, 0.5], c=0.1)
