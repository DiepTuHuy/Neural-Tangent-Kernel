"""
==============================================================================
PHENOMENON 2: Time-Varying NTK and Spectral Evolution
==============================================================================

Requires: torch, numpy, matplotlib
Run:      python phenomenon2_ntk_spectral.py

What this script demonstrates:
  ① The NTK as an inner product of Jacobians: K(x,x') = ∇_θf(x)ᵀ ∇_θf(x')
  ② How NTK eigenvalues λₖ set the convergence rate of each error mode
  ③ How ω(t) reshapes the NTK spectrum during training:
       K_ω ∝ ω²   →  increasing ω inflates high-frequency eigenvalues
  ④ Four omega schedules: constant | linear | exponential | warmup-decay
  ⑤ NTK condition number as a stability proxy

Signal:
  f(x) = sin(2πx) + 0.5·sin(10πx) + 0.2·sin(30πx)

Outputs:
  phenomenon2_ntk_spectra.png      — eigenvalue spectra + condition numbers
  phenomenon2_spectral_dynamics.png — spectral heatmaps + loss + ω(t) curves
  phenomenon2_reconstructions.png   — final predictions + FFT per schedule

NTK computation:
  Uses torch.autograd.functional.jacobian — exact, not finite-difference.
  A small grid (n=20) is used so the n×n NTK is tractable; this is
  enough to capture the eigenvalue structure.
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIREN  (same architecture as Phenomenon 1)
# ═════════════════════════════════════════════════════════════════════════════

class SIRENLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int,
                 omega: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega  = omega
        self.linear = nn.Linear(in_f, out_f)
        fan = self.linear.in_features
        with torch.no_grad():
            b = 1.0/fan if is_first else (6.0/fan)**0.5 / omega
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-torch.pi, torch.pi)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

    def set_omega(self, w):
        self.omega = w


class SIREN(nn.Module):
    """
    SIREN network.  Smaller hidden_dim (64) and fewer layers (3 hidden)
    compared to Phenomenon 1 so that the exact NTK Jacobian is tractable.

    Args:
        in_f:    input dim
        hid:     hidden width
        n_hid:   number of hidden SIREN layers
        out_f:   output dim
        omega:   frequency scaling ω
    """
    def __init__(self, in_f=1, hid=64, n_hid=3, out_f=1, omega=30.0):
        super().__init__()
        self.omega = omega
        layers = [SIRENLayer(in_f, hid, omega=omega, is_first=True)]
        for _ in range(n_hid - 1):
            layers.append(SIRENLayer(hid, hid, omega=omega, is_first=False))
        self.hidden = nn.ModuleList(layers)
        self.head   = nn.Linear(hid, out_f)
        with torch.no_grad():
            b = (6.0/hid)**0.5 / omega
            self.head.weight.uniform_(-b, b)
            self.head.bias.zero_()

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = layer(h)
        return self.head(h)

    def set_omega(self, omega):
        self.omega = omega
        for layer in self.hidden:
            layer.set_omega(omega)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIGNAL AND DATASET
# ═════════════════════════════════════════════════════════════════════════════

def target_signal(x: np.ndarray) -> np.ndarray:
    return (np.sin(2*np.pi*x)
            + 0.5*np.sin(10*np.pi*x)
            + 0.2*np.sin(30*np.pi*x))


def make_dataset(n: int = 512):
    x = np.linspace(0, 1, n, dtype=np.float32)
    y = target_signal(x).astype(np.float32)
    X = torch.from_numpy(x).unsqueeze(-1).to(DEVICE)
    Y = torch.from_numpy(y).unsqueeze(-1).to(DEVICE)
    return X, Y


def eval_grid(n: int = 512) -> torch.Tensor:
    return torch.linspace(0, 1, n).unsqueeze(-1).to(DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NTK COMPUTATION  (exact Jacobian via torch.autograd)
# ═════════════════════════════════════════════════════════════════════════════

def compute_jacobian(model: SIREN, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the exact Jacobian  J[i,k] = ∂f(xᵢ)/∂θₖ
    using torch.autograd's backward pass per sample.

    For each input xᵢ we do one backward pass and concatenate the
    parameter gradients into row i of J.

    Returns J of shape  (n_inputs, n_params).

    Note: This uses the scalar-output assumption (out_features=1).
    For vector outputs you would need vmap or functional.jacobian.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    n        = x.shape[0]
    J        = torch.zeros(n, n_params, device=x.device)

    for i in range(n):
        model.zero_grad()
        xi   = x[i:i+1].clone()                # (1, in_f)
        out  = model(xi)                        # (1, 1)
        out.backward(torch.ones_like(out))      # scalar output → grad
        grads = []
        for p in params:
            g = p.grad.detach().view(-1) if p.grad is not None \
                else torch.zeros(p.numel(), device=x.device)
            grads.append(g)
        J[i] = torch.cat(grads)

    return J                                    # (n, n_params)


def compute_ntk(model: SIREN, x: torch.Tensor) -> torch.Tensor:
    """
    NTK ≈ J(x) · J(x)ᵀ     shape (n, n)

    K[i,j] = <∇_θ f(xᵢ), ∇_θ f(xⱼ)>

    Interpretation:
      • K encodes how similar two points are in parameter-gradient space.
      • Its eigenvalues λₖ control the convergence rate of each error mode:
            eₖ(t) = exp(−λₖ · t) · eₖ(0)
      • Large λₖ → fast convergence (typically low-frequency modes).
      • Small λₖ → slow convergence (typically high-frequency modes).
      • Increasing ω scales every Jacobian entry by ω (via chain rule through
        the cos(·) derivative), so K ∝ ω² and all eigenvalues inflate.
    """
    J = compute_jacobian(model, x)              # (n, n_params)
    return (J @ J.T).detach().cpu()             # (n, n)


def ntk_eigenvalues(model: SIREN, x: torch.Tensor) -> np.ndarray:
    """Return NTK eigenvalues in descending order as numpy array."""
    K    = compute_ntk(model, x).numpy()
    eigs = np.linalg.eigvalsh(K)               # ascending (symmetric solver)
    return eigs[::-1].copy()                    # descending


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OMEGA SCHEDULES
# ═════════════════════════════════════════════════════════════════════════════

def sched_constant(val):
    """ω(t) = val (constant)."""
    return lambda it, T: val


def sched_linear(w0, w1):
    """ω(t) = w0 + (w1-w0)·t/T   (linear ramp from w0 to w1)."""
    return lambda it, T: w0 + (w1 - w0) * it / max(T - 1, 1)


def sched_exponential(w0, w1):
    """ω(t) = w0·(w1/w0)^(t/T)   (geometric / exponential ramp).

    Mathematical connection to NTK:
      Since J_ω ∝ ω·cos(ωWx), K_ω = J_ω·J_ωᵀ ∝ ω².
      Exponential growth of ω therefore causes quadratic growth of every
      NTK eigenvalue at each training step — rapidly unlocking high-freq modes.
    """
    return lambda it, T: w0 * (w1 / w0) ** (it / max(T - 1, 1))


def sched_warmup_decay(w_peak):
    """
    Curriculum schedule:
      Phase 1 (0→T/2): ramp from 5 to w_peak  (add detail progressively)
      Phase 2 (T/2→T): ramp from w_peak down to w_peak/3  (stabilise)

    The decay phase reduces the condition number, improving late-stage
    stability for high-frequency components.
    """
    def fn(it, T):
        ph = it / max(T - 1, 1)
        if ph < 0.5:
            return 5.0 + (w_peak - 5.0) * (2 * ph)
        else:
            return w_peak - (w_peak - w_peak / 3) * 2 * (ph - 0.5)
    return fn


SCHEDULES = {
    'Constant ω=30':    (sched_constant(30.0),        30.0),
    'Linear ω: 5→60':  (sched_linear(5.0, 60.0),      5.0),
    'Exp ω: 5→60':     (sched_exponential(5.0, 60.0),  5.0),
    'Warmup-decay ω':  (sched_warmup_decay(50.0),       5.0),
}

SCHED_C = {
    'Constant ω=30':   '#60A5FA',
    'Linear ω: 5→60':  '#34D399',
    'Exp ω: 5→60':     '#F472B6',
    'Warmup-decay ω':  '#FBBF24',
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING WITH REAL AUTOGRAD + SNAPSHOT COLLECTION
# ═════════════════════════════════════════════════════════════════════════════

def train_with_schedule(model: SIREN, X: torch.Tensor, Y: torch.Tensor,
                        n_epochs: int,
                        sched_fn,               # callable(it, T) → float
                        snapshot_epochs: list,
                        lr: float = 1e-4,
                        print_every: int = 500):
    """
    PyTorch Adam training loop with:
      • Live ω(t) updates   (model.set_omega called each epoch)
      • Snapshot saving      (model state_dict at requested epochs)
      • MSE loss tracking

    At each snapshot we record:
      state_dict : full model weights (for NTK computation)
      omega      : current ω value
      pred       : prediction on the evaluation grid (for FFT heatmap)

    Returns:
      losses    : list of MSE per epoch
      snapshots : dict { epoch: {'state': ..., 'omega': ..., 'pred': ...} }
    """
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_eval  = eval_grid(512)

    losses    = []
    snapshots = {}

    model.train()
    for ep in range(n_epochs):
        # Update ω
        new_omega = sched_fn(ep, n_epochs)
        model.set_omega(new_omega)

        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()        # ← real autograd
        opt.step()
        losses.append(loss.item())

        if ep in snapshot_epochs:
            with torch.no_grad():
                pred_snap = model(x_eval).squeeze().cpu().numpy()
            snapshots[ep] = {
                'state': deepcopy(model.state_dict()),
                'omega': model.omega,
                'pred' : pred_snap,
            }

        if (ep + 1) % print_every == 0:
            print(f"    ep {ep+1:5d}/{n_epochs}  "
                  f"ω={model.omega:6.2f}  MSE={loss.item():.6f}")

    return losses, snapshots


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(X).squeeze().cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FFT
# ═════════════════════════════════════════════════════════════════════════════

def compute_fft(signal, fs=512.0):
    n   = len(signal)
    fft = np.fft.rfft(signal)
    amp = (2.0/n)*np.abs(fft); amp[0] /= 2
    frq = np.fft.rfftfreq(n, d=1.0/fs)
    return frq, amp


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN EXPERIMENT LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_experiments():
    print("=" * 62)
    print("PHENOMENON 2 — Time-Varying NTK & Spectral Evolution  [PyTorch]")
    print("=" * 62)

    X_train, Y_train = make_dataset(512)
    x_eval           = eval_grid(512)
    y_gt             = target_signal(x_eval.squeeze().cpu().numpy())

    # Small grid for NTK (n=20 → 20×20 matrix, tractable exact Jacobian)
    x_ntk = torch.linspace(0, 1, 20).unsqueeze(-1).to(DEVICE)

    N_EPOCHS      = 3000
    SNAP_EPOCHS   = sorted({0, 300, 600, 1000, 1500, 2000, 2999})
    HID, N_HID    = 64, 3         # smaller model for tractable NTK

    all_results = {}

    for name, (sched_fn, omega_init) in SCHEDULES.items():
        print(f"\n{'─'*55}\n[Schedule] {name}")
        torch.manual_seed(42)
        model = SIREN(in_f=1, hid=HID, n_hid=N_HID,
                      out_f=1, omega=omega_init).to(DEVICE)

        losses, snaps = train_with_schedule(
            model, X_train, Y_train,
            n_epochs    = N_EPOCHS,
            sched_fn    = sched_fn,
            snapshot_epochs = SNAP_EPOCHS,
            lr          = 1e-4,
            print_every = 600,
        )

        final_pred = predict(model, x_eval)

        # ── Compute exact NTK eigenvalues at each snapshot ────────────────────
        print(f"  Computing NTK eigenvalues at {len(SNAP_EPOCHS)} snapshots ...")
        eigs_time = {}
        ntk_mats  = {}

        for ep, snap in sorted(snaps.items()):
            # Restore snapshot into temporary model
            tmp = SIREN(in_f=1, hid=HID, n_hid=N_HID,
                        out_f=1, omega=snap['omega']).to(DEVICE)
            tmp.load_state_dict(snap['state'])
            tmp.set_omega(snap['omega'])

            eigs      = ntk_eigenvalues(tmp, x_ntk)
            K         = compute_ntk(tmp, x_ntk).numpy()
            eigs_time[ep] = eigs
            ntk_mats[ep]  = K

            lmax = eigs[0]; lmin = eigs[-1]
            cond = lmax / (lmin + 1e-12)
            print(f"    ep {ep:4d} | ω={snap['omega']:6.2f} "
                  f"| λ_max={lmax:.3e}  λ_min={lmin:.3e}  cond={cond:.2e}")

        all_results[name] = {
            'losses'    : losses,
            'snaps'     : snaps,
            'final_pred': final_pred,
            'eigs'      : eigs_time,
            'ntk_mats'  : ntk_mats,
            'sched_fn'  : sched_fn,
            'omega_init': omega_init,
        }

    return all_results, x_eval.squeeze().cpu().numpy(), y_gt, N_EPOCHS


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CONSOLE THEORY SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_ntk_theory(all_results):
    print("\n" + "─" * 62)
    print("NTK THEORY SUMMARY")
    print("─" * 62)
    print("""
Definition:
  K(x,x';θ) = ∇_θf(x)ᵀ · ∇_θf(x')  ≈  J(x)·J(x')ᵀ

Gradient flow dynamics (MSE loss):
  de/dt = −K(X,X;θ)·e(t)

Eigendecomposition:
  K = Q·diag(λ₁,…,λₙ)·Qᵀ
  eₖ(t) = exp(−λₖ·t)·eₖ(0)
  → mode k converges at rate λₖ
  → large λₖ = fast  (typically low-frequency)
  → small λₖ = slow  (typically high-frequency)

Effect of ω on K:
  ∂f/∂W ∝ ω·cos(ωWx)    [chain rule through sin]
  K_ω = J_ω·J_ωᵀ  ∝  ω²
  → Doubling ω quadruples all eigenvalues
  → High-freq eigenmodes converge proportionally faster
  → But condition number can blow up → instability
""")
    print("Numerical results — condition number at final snapshot:")
    for name, res in all_results.items():
        eigs = res['eigs']
        it_f = max(eigs.keys())
        ev   = eigs[it_f]
        cond = ev[0] / (ev[-1] + 1e-12)
        print(f"  {name:<22s}: λ_max={ev[0]:.3e}  "
              f"λ_min={ev[-1]:.3e}  cond={cond:.2e}")
    print("─" * 62)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

C_GT = '#2563EB'


def _ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#1E293B')
    for sp in ax.spines.values(): sp.set_edgecolor('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_xlabel(xlabel, color='#CBD5E1', fontsize=10)
    ax.set_ylabel(ylabel, color='#CBD5E1', fontsize=10)
    ax.set_title(title,   color='#F1F5F9', fontsize=11, pad=8)
    ax.grid(True, color='#1E3A5F', lw=0.5, alpha=0.7)


def _leg(ax, loc='best'):
    ax.legend(fontsize=8, facecolor='#0F172A',
              labelcolor='#CBD5E1', edgecolor='#334155', loc=loc)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — NTK eigenvalue spectra + condition numbers
# ─────────────────────────────────────────────────────────────────────────────

def plot_ntk_spectra(all_results):
    """
    2-row figure:
      Row 0: NTK eigenvalue spectrum at each training snapshot (per schedule)
      Row 1 left:   condition number λ_max/λ_min over time (all schedules)
      Row 1 mid:    λ_max trajectory over time (all schedules)
      Row 1 right:  top-5 eigenvalue trajectories for the exp schedule
      Row 1 far-r:  NTK matrix heatmap at final epoch (exp schedule)
    """
    names  = list(all_results.keys())
    n      = len(names)
    plasma = plt.get_cmap('plasma')

    fig = plt.figure(figsize=(5 * n + 2, 12))
    fig.patch.set_facecolor('#0F172A')
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.48, wspace=0.34)

    # ── Row 0: spectrum per schedule ──────────────────────────────────────────
    for col, name in enumerate(names):
        res   = all_results[name]
        eigs  = res['eigs']
        iters = sorted(eigs.keys())
        ax    = fig.add_subplot(gs[0, col])
        for i, it in enumerate(iters):
            ev    = np.abs(eigs[it]) + 1e-12
            shade = plasma(i / max(len(iters) - 1, 1))
            ax.semilogy(np.arange(1, len(ev)+1), ev,
                        color=shade, lw=1.4, alpha=0.85,
                        label=f'ep {it}')
        _ax(ax, f'NTK spectrum\n{name}',
            'Eigenvalue index (desc)', '|λ| (log)')
        _leg(ax, loc='upper right')

    # ── Row 1: shared diagnostics across all schedules ────────────────────────
    # condition number
    ax_c = fig.add_subplot(gs[1, 0])
    for name in names:
        eigs  = all_results[name]['eigs']
        iters = sorted(eigs.keys())
        conds = [eigs[i][0] / (eigs[i][-1] + 1e-12) for i in iters]
        ax_c.semilogy(iters, conds, color=SCHED_C[name],
                      lw=2.0, marker='o', ms=5, label=name)
    _ax(ax_c, 'NTK condition number  λ_max/λ_min',
        'Epoch', 'Condition no. (log)')
    _leg(ax_c)

    # lambda_max
    ax_lm = fig.add_subplot(gs[1, 1])
    for name in names:
        eigs  = all_results[name]['eigs']
        iters = sorted(eigs.keys())
        lmaxs = [eigs[i][0] for i in iters]
        ax_lm.semilogy(iters, lmaxs, color=SCHED_C[name],
                       lw=2.0, marker='s', ms=5, label=name)
    _ax(ax_lm, 'λ_max over training', 'Epoch', 'λ_max (log)')
    _leg(ax_lm)

    # top-k eigenvalue trajectories for exp schedule
    ax_tk = fig.add_subplot(gs[1, 2])
    name_e = 'Exp ω: 5→60'
    eigs_e = all_results[name_e]['eigs']
    iters_e= sorted(eigs_e.keys())
    kc     = ['#F59E0B','#10B981','#60A5FA','#F472B6','#A78BFA']
    ntop   = min(5, len(list(eigs_e.values())[0]))
    for k in range(ntop):
        vals = [eigs_e[i][k] for i in iters_e]
        ax_tk.semilogy(iters_e, vals, color=kc[k], lw=1.8,
                       marker='o', ms=4, label=f'λ_{k+1}')
    _ax(ax_tk, f'Top-{ntop} eigenvalues\n[Exp ω schedule]',
        'Epoch', 'λ (log)')
    _leg(ax_tk)

    # NTK matrix heatmap at final epoch for exp schedule
    ax_hm = fig.add_subplot(gs[1, 3])
    ntk_mats = all_results[name_e]['ntk_mats']
    it_f     = max(ntk_mats.keys())
    K_f      = ntk_mats[it_f]
    im = ax_hm.imshow(np.log10(np.abs(K_f) + 1e-10),
                      cmap='magma', aspect='auto', interpolation='nearest')
    ax_hm.set_title(f'NTK matrix K[i,j]\n[Exp ω, ep {it_f}]',
                    color='#F1F5F9', fontsize=11, pad=8)
    ax_hm.set_xlabel('Input j', color='#CBD5E1', fontsize=10)
    ax_hm.set_ylabel('Input i', color='#CBD5E1', fontsize=10)
    ax_hm.tick_params(colors='#94A3B8', labelsize=9)
    for sp in ax_hm.spines.values(): sp.set_edgecolor('#334155')
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb.set_label('log₁₀|K(xᵢ,xⱼ)|', color='#94A3B8', fontsize=8)
    cb.ax.tick_params(colors='#94A3B8', labelsize=8)

    fig.suptitle(
        'Phenomenon 2 — NTK Eigenvalue Spectra Under Different ω(t)  '
        '[Exact Jacobians via PyTorch autograd]',
        color='#F8FAFC', fontsize=13, fontweight='bold', y=0.995)

    out = 'phenomenon2_ntk_spectra.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Spectral evolution heatmaps + loss + ω(t) curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_dynamics(all_results, y_gt, N_EPOCHS, fs=512.0):
    """
    2-row figure:
      Row 0: spectral heatmap (frequency × epoch) per schedule
             Yellow horizontal lines = true signal frequencies (1, 5, 15 Hz)
             Colour = log₁₀(amplitude)  — shows when each frequency is learned
      Row 1: MSE loss (left) + ω(t) curves for the 3 dynamic schedules
    """
    names  = list(all_results.keys())
    n      = len(names)
    fg_gt, ag_gt = compute_fft(y_gt, fs)
    max_f  = 60.0
    mk     = fg_gt <= max_f

    fig = plt.figure(figsize=(5 * n + 2, 11))
    fig.patch.set_facecolor('#0F172A')
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.48, wspace=0.34)

    # ── Row 0: spectral heatmaps ──────────────────────────────────────────────
    for col, name in enumerate(names):
        res   = all_results[name]
        snaps = res['snaps']
        iters = sorted(snaps.keys())
        amp_matrix = np.array([
            compute_fft(snaps[i]['pred'], fs)[1][mk]
            for i in iters
        ])
        amp_matrix = np.clip(amp_matrix, 1e-6, None)

        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            np.log10(amp_matrix).T,
            aspect='auto', origin='lower',
            extent=[iters[0], iters[-1], fg_gt[mk][0], fg_gt[mk][-1]],
            cmap='inferno', interpolation='bilinear',
            vmin=-4, vmax=0,
        )
        for ft, lb in [(1,'1 Hz'),(5,'5 Hz'),(15,'15 Hz')]:
            ax.axhline(ft, color='#FACC15', lw=1.2, ls='--', alpha=0.9)
            ax.text(iters[-1] * 0.97, ft + 0.5, lb,
                    color='#FACC15', fontsize=8, ha='right')
        ax.set_title(f'Spectral evolution\n{name}',
                     color='#F1F5F9', fontsize=11, pad=8)
        ax.set_xlabel('Epoch',          color='#CBD5E1', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', color='#CBD5E1', fontsize=10)
        ax.tick_params(colors='#94A3B8', labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#334155')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('log₁₀(Amp)', color='#94A3B8', fontsize=8)
        cb.ax.tick_params(colors='#94A3B8', labelsize=8)

    # ── Row 1, col 0: loss curves ─────────────────────────────────────────────
    ax_l = fig.add_subplot(gs[1, 0])
    for name in names:
        ax_l.semilogy(all_results[name]['losses'],
                      color=SCHED_C[name], lw=2.0, label=name)
    _ax(ax_l, 'MSE Loss — all ω schedules  [Adam lr=1e-4]',
        'Epoch', 'MSE (log)')
    _leg(ax_l)

    # ── Row 1, cols 1-3: ω(t) trajectories ───────────────────────────────────
    dyn_names = [n for n in names if n != 'Constant ω=30']
    iters_all = np.arange(N_EPOCHS)
    for c, name in enumerate(dyn_names, start=1):
        fn     = all_results[name]['sched_fn']
        omegas = [fn(i, N_EPOCHS) for i in iters_all]
        ax     = fig.add_subplot(gs[1, c])
        ax.plot(iters_all, omegas, color=SCHED_C[name], lw=2.5)
        ax.fill_between(iters_all, omegas, alpha=0.15, color=SCHED_C[name])
        _ax(ax, f'ω(t) schedule\n{name}', 'Epoch', 'ω')
        ax.set_ylim(bottom=0)

    fig.suptitle(
        'Phenomenon 2 — Spectral Evolution Heatmaps & ω(t) Schedules  '
        '[PyTorch]',
        color='#F8FAFC', fontsize=13, fontweight='bold', y=0.995)

    out = 'phenomenon2_spectral_dynamics.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Final reconstructions + FFT per schedule
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstructions(all_results, x_eval, y_gt, fs=512.0):
    """
    2-row figure:
      Row 0: final signal reconstruction for each omega schedule
      Row 1: FFT of final prediction vs GT for each schedule
    """
    names  = list(all_results.keys())
    n      = len(names)
    fg_gt, ag_gt = compute_fft(y_gt, fs)
    max_f  = 60.0
    mk     = fg_gt <= max_f

    fig = plt.figure(figsize=(5 * n + 2, 10))
    fig.patch.set_facecolor('#0F172A')
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.34)

    for col, name in enumerate(names):
        color = SCHED_C[name]
        pred  = all_results[name]['final_pred']
        _, ap = compute_fft(pred, fs)
        rmse  = np.sqrt(np.mean((pred - y_gt) ** 2))

        # reconstruction
        ax = fig.add_subplot(gs[0, col])
        ax.plot(x_eval, y_gt, color=C_GT,   lw=2.0, label='Ground truth')
        ax.plot(x_eval, pred, color=color,  lw=1.5, ls='--', label=name)
        _ax(ax, f'{name}\nRMSE = {rmse:.4f}', 'x', 'f(x)')
        _leg(ax)

        # FFT
        ax = fig.add_subplot(gs[1, col])
        ax.fill_between(fg_gt[mk], ag_gt[mk], alpha=0.20, color=C_GT)
        ax.fill_between(fg_gt[mk], ap[mk],     alpha=0.20, color=color)
        ax.plot(fg_gt[mk], ag_gt[mk], color=C_GT,  lw=2.0, label='GT')
        ax.plot(fg_gt[mk], ap[mk],    color=color,  lw=1.5, ls='--',
                label='Prediction')
        for ft in [1, 5, 15]:
            ax.axvline(ft, color='#FACC15', lw=0.8, ls=':', alpha=0.75)
        _ax(ax, f'FFT — {name}', 'Frequency (Hz)', 'Amplitude')
        _leg(ax)

    fig.suptitle(
        'Phenomenon 2 — Signal Reconstructions & FFT Under Different ω(t)  '
        '[PyTorch]',
        color='#F8FAFC', fontsize=13, fontweight='bold', y=0.995)

    out = 'phenomenon2_reconstructions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    all_results, x_eval, y_gt, N_EPOCHS = run_experiments()

    print_ntk_theory(all_results)

    print("\nGenerating figures ...")
    plot_ntk_spectra(all_results)
    plot_spectral_dynamics(all_results, y_gt, N_EPOCHS)
    plot_reconstructions(all_results, x_eval, y_gt)

    print("\n✓  Phenomenon 2 complete.")
    print("   → phenomenon2_ntk_spectra.png")
    print("   → phenomenon2_spectral_dynamics.png")
    print("   → phenomenon2_reconstructions.png")
