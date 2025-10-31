# Simulation of dynamics of egg shape formation — generates GIF for egg shape contour
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

# ---------- Preset R1 ----------
a = 1.0
b0, b1 = 1.0, 0.62
u = 1.0
h = 0.005
alpha_target = 0.001
alpha_cap = 0.90

# ---------- Dynamics ----------
gamma = 1.5
dt = 0.04
n_frames = 160
lock_last_k = 20

theta = np.linspace(0, 2*np.pi, 900)
dpi = 120
frame_duration = 0.06
gif_path = "egg_contour.gif"

def cardano_real_roots(r, u, h):
    p = r / u
    q = -h / u
    Delta = (q/2)**2 + (p/3)**3
    if Delta > 0:
        A = np.cbrt(-q/2 + np.sqrt(Delta))
        B = np.cbrt(-q/2 - np.sqrt(Delta))
        return [A + B]
    if abs(p) < 1e-14:
        return [np.cbrt(-q)]
    if p < 0:
        rho = 2*np.sqrt(-p/3)
        acos_arg = (3*q/(2*p))*np.sqrt(-3/p)
        acos_arg = np.clip(acos_arg, -1.0, 1.0)
        phi = np.arccos(acos_arg)
        return [rho*np.cos((phi - 2*np.pi*k)/3) for k in range(3)]
    return [np.cbrt(-q)]

def stable_alpha_star(r, u, h, alpha_prev):
    roots = cardano_real_roots(r, u, h)
    stable = [a for a in roots if (r + 3*u*a*a) > 0]
    if not stable:
        stable = roots
    return min(stable, key=lambda x: abs(x - alpha_prev))

def curve_bounds(a, b, alpha_samples):
    xs, ys = [], []
    th = np.linspace(0, 2*np.pi, 900)
    for al in alpha_samples:
        denom = 1 + al * np.cos(th)
        denom = np.where(np.abs(denom)<1e-3, np.sign(denom)*1e-3, denom)
        x = a * np.cos(th)
        y = b * np.sin(th) / denom
        xs.append(x); ys.append(y)
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    pad = 0.2
    xmin, xmax = xs.min()-pad, xs.max()+pad
    ymin, ymax = ys.min()-pad, ys.max()+pad
    m = max(xmax-xmin, ymax-ymin)
    cx = 0.5*(xmin+xmax); cy = 0.5*(ymin+ymax)
    return cx-0.55*m, cx+0.55*m, cy-0.55*m, cy+0.55*m

# Derived r-values
r0 = 2.0
r1 = (h - u*alpha_target**3) / alpha_target
r_sched = np.linspace(r0, r1, n_frames)
b_sched = np.linspace(b0, b1, n_frames)

alpha_preview = np.linspace(0.0, alpha_target, 18)
xmin, xmax, ymin, ymax = curve_bounds(a, b1, alpha_preview)

images = []
fig = plt.figure(figsize=(6.6, 6.6), dpi=dpi)
alpha = 0.0
alpha_star_prev = 0.0

for i in range(n_frames):
    r = float(r_sched[i])
    b = float(b_sched[i])
    a_star = stable_alpha_star(r, u, h, alpha_star_prev)
    alpha_star_prev = a_star

    dF_dalpha = r * alpha + u * alpha**3 - h
    alpha = alpha - gamma * dF_dalpha * dt

    if i >= n_frames - lock_last_k:
        alpha = a_star

    alpha = float(np.clip(alpha, -alpha_cap, alpha_cap))

    denom = 1 + alpha * np.cos(theta)
    denom = np.where(np.abs(denom)<1e-3, np.sign(denom)*1e-3, denom)
    x = a * np.cos(theta)
    y = b * np.sin(theta) / denom

    plt.clf()
    plt.plot(x, y, linewidth=3, color='red')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xticks([]); plt.yticks([])
    #plt.title("Egg Shape Formation")

    txt = (f"frame {i+1}/{n_frames}\n"
           f"r(t)={r:.3f},  b(t)={b:.2f}\n"
           f"α={alpha:.3f},  α*(r)={a_star:.3f}\n"
           f"target α*={alpha_target:.3f},  r1={r1:.3f}\n"
           f"b0={b0}, b1={b1}")
    plt.text(0.02, 0.02, txt, transform=plt.gca().transAxes, fontsize=9,
             ha='left', va='bottom')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    images.append(imageio.v2.imread(buf))
    buf.close()

plt.close(fig)
imageio.mimsave(gif_path, images, duration=frame_duration)
gif_path
