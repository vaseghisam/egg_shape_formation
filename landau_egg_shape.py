import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

# ---------------- Scenario parameters ----------------
u = 1.0
h = 0.005
alpha_target = 0.2

# r ramp: start symmetric (r0>0), end broken symmetry (r1<0)
r0 = 2.0
r1 = (h - u * alpha_target**3) / alpha_target  # expected: -0.015

# ---------------- Animation settings ----------------
n_ramp_frames = 120          # frames while r ramps from r0 to r1
n_hold_frames = 24           # extra frames at r = r1 to show final equilibrium
duration = 0.06              # seconds per frame
alpha_min, alpha_max = -1.1, 1.1
alpha_grid = np.linspace(alpha_min, alpha_max, 2000)

# Strict y-range for clarity (shifted potential)
Y_LO, Y_HI = 0.0, 0.003

# ---------------- Landau model helpers ----------------
def F(alpha, r, u, h):
    """Landau potential: F(α) = 1/2 r α^2 + 1/4 u α^4 − h α."""
    return 0.5 * r * alpha**2 + 0.25 * u * alpha**4 - h * alpha

def cubic_real_roots(r, u, h):
    """
    Solve u α^3 + r α − h = 0 for real roots.
    Returns a list of real roots (1 or 3 elements depending on regime).
    """
    # Normalize: x^3 + p x + q = 0
    p = r / u
    q = -h / u
    Delta = (q / 2)**2 + (p / 3)**3

    if Delta > 0:
        # One real root
        A = np.cbrt(-q/2 + np.sqrt(Delta))
        B = np.cbrt(-q/2 - np.sqrt(Delta))
        return [A + B]

    if abs(p) < 1e-14:
        # x^3 + q = 0
        return [np.cbrt(-q)]

    # Three real roots (trigonometric form) when p < 0
    if p < 0:
        rho = 2 * np.sqrt(-p / 3)
        acos_arg = (3 * q / (2 * p)) * np.sqrt(-3 / p)
        acos_arg = np.clip(acos_arg, -1.0, 1.0)
        phi = np.arccos(acos_arg)
        return [rho * np.cos((phi - 2 * np.pi * k) / 3) for k in range(3)]

    # Fallback: treat as single real root
    return [np.cbrt(-q)]

def stationary_points(r, u, h):
    """
    Return (roots, stability), where stability = +1 for minima, -1 for maxima,
    using sign of second derivative: F''(α) = r + 3 u α^2.
    """
    roots = cubic_real_roots(r, u, h)
    st = []
    for a in roots:
        sec = r + 3 * u * a * a
        st.append(1 if sec > 0 else -1)
    return roots, st

# ---------------- r schedule (ramp + hold) ----------------
ramp = np.linspace(r0, r1, n_ramp_frames)
hold = np.full(n_hold_frames, r1)
r_sched = np.concatenate([ramp, hold])
n_frames = r_sched.size

# ---------------- Render animation ----------------
images = []
fig = plt.figure(figsize=(7.0, 5.2), dpi=110)  # fixed canvas size for consistent frame dimensions

for i in range(n_frames):
    r = float(r_sched[i])

    # Compute potential and shift so min is at 0 each frame
    Fa = F(alpha_grid, r, u, h)
    Fmin = float(Fa.min())
    Fa_shift = Fa - Fmin

    plt.clf()
    plt.plot(alpha_grid, Fa_shift, linewidth=2)
    # plt.xlim(alpha_min, alpha_max)
    plt.xlim(-0.5, 0.5)
    plt.ylim(Y_LO, Y_HI)
    plt.xlabel("α")
    plt.ylabel("F(α) − min F(α)")
    #plt.title("Landau Potential Across Phase Transition and Egg Shape Formation")

    # Mark stationary points: • minima, × maximum (barrier)
    roots, st = stationary_points(r, u, h)
    # Sort left-to-right for visual consistency
    idx = np.argsort(roots)
    roots = [roots[j] for j in idx]
    st = [st[j] for j in idx]

    for a, sgn in zip(roots, st):
        Fa_pt = F(a, r, u, h) - Fmin
        if sgn > 0:
            plt.scatter([a], [Fa_pt], s=55)             # stable min (•)
        else:
            plt.scatter([a], [Fa_pt], s=55, marker="x")  # unstable max (×)

    # Diagnostics text
    txt = (
        f"frame {i+1}/{n_frames}   r={r:.4f}\n"
        f"u={u:.2f},  h={h:.3f}\n"
        f"target α*={alpha_target:.3f},  r1={r1:.3f}"
    )
    plt.text(0.02, 0.02, txt, transform=plt.gca().transAxes, fontsize=9,
             ha='left', va='bottom')

    buf = BytesIO()
    # IMPORTANT: keep default bbox so every frame has identical pixel dimensions
    plt.savefig(buf, format='png', dpi=110)
    buf.seek(0)
    images.append(imageio.v2.imread(buf))
    buf.close()

plt.close(fig)

# Save GIF
gif_path = "landau_phase_transition.gif"
imageio.mimsave(gif_path, images, duration=duration)

print("Saved:", gif_path)
