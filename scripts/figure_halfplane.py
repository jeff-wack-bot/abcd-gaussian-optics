"""Figure: Gaussian beams as points of the hyperbolic half-plane.
Plotted coordinate: tau = conj(q) = z + i zR  (z = distance past the waist, zR = Rayleigh range),
i.e. Siegman/Kogelnik's q.  Real ABCD matrices act on tau by the same Mobius map."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#888888"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})

def mob(M, t): return (M[0][0] * t + M[0][1]) / (M[1][0] * t + M[1][1])
def F(d): return np.array([[1, d], [0, 1.0]])
def Lf(f): return np.array([[1, 0], [-1 / f, 1.0]])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))

# (a) path through free space / lens / free space / lens / free space
t = 0.0 + 1.0j   # units: metres, start at a waist with zR = 1
path = []
def go_free(t, d, n=200):
    s = np.linspace(0, d, n); seg = t + s; path.append(("free", seg)); return seg[-1]
def go_lens(t, f, n=200):
    s = np.linspace(0, 1, n); seg = 1 / (1 / t - s / f); path.append(("lens", seg)); return seg[-1]
t = go_free(t, 1.0); t = go_lens(t, 0.8); t = go_free(t, 1.6); t = go_lens(t, -1.5); t = go_free(t, 1.0)
for kind, seg in path:
    a1.plot(seg.real, seg.imag, color=BLUE if kind == "free" else ORANGE, lw=2,
            solid_capstyle="round")
# faint lens horocycles through the start of each lens segment (circles tangent to the axis at 0)
for kind, seg in path:
    if kind == "lens":
        c = 1 / seg[0]; s = np.linspace(-40, 40, 4000)
        h = 1 / (c - s)
        a1.plot(h.real, h.imag, color=ORANGE, lw=0.6, alpha=0.35, zorder=0)
a1.plot([path[0][1][0].real], [path[0][1][0].imag], "o", ms=5, color=INK)
a1.annotate("input waist", (0, 1), (0.15, 1.25), fontsize=8, color=INK)
a1.plot([t.real], [t.imag], "o", ms=5, color=INK)
a1.annotate("output", (t.real, t.imag), (t.real - 0.55, t.imag + 0.3), fontsize=8, color=INK)
a1.plot([], [], color=BLUE, lw=2, label="free space: $\\tau\\mapsto\\tau+d$")
a1.plot([], [], color=ORANGE, lw=2, label="thin lens: $1/\\tau\\mapsto 1/\\tau-1/f$")
a1.axhline(0, color=MUTED, lw=0.8)
a1.set_xlim(-1.1, 2.3); a1.set_ylim(0, 2.6); a1.set_aspect("equal")
a1.set_xlabel("Re $\\tau$ = distance past waist $z$ (m)"); a1.set_ylabel("Im $\\tau$ = Rayleigh range $z_R$ (m)")
a1.set_title("(a) a lens system moves a point of $\\mathbb{H}$", fontsize=9, color=INK)
a1.legend(loc="upper left", fontsize=7.5, frameon=False)
a1.spines[["top", "right"]].set_visible(False)

# (b) stable resonator round trips: eigenmode = fixed point, mismatched beam circles it
R1, R2, L = 2.0, 3.0, 1.2
Mrt = Lf(R1 / 2) @ F(L) @ Lf(R2 / 2) @ F(L)
A, B, C, D = Mrt.ravel()
fp = ((A - D) + np.sqrt(complex((A + D) ** 2 - 4))) / (2 * C)
fp = fp if fp.imag > 0 else np.conj(fp)
for r, col in [(0.35, BLUE), (0.8, AQUA)]:
    # hyperbolic circle of radius r about fp: Euclidean circle centre (x, y cosh r), radius y sinh r
    th = np.linspace(0, 2 * np.pi, 400)
    cx, cy, rr = fp.real, fp.imag * np.cosh(r), fp.imag * np.sinh(r)
    a2.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), color=col, lw=0.8, alpha=0.5)
    t = cx + 1j * (cy + rr)
    pts = [t]
    for _ in range(14):
        t = mob(Mrt, t); pts.append(t)
    pts = np.array(pts)
    a2.plot(pts.real, pts.imag, "o", ms=4, color=col, mec="white", mew=0.8,
            label=f"mismatched beam, $d_{{\\mathbb{{H}}}}={r}$")
a2.plot([fp.real], [fp.imag], "*", ms=10, color=INK, label="eigenmode (fixed point)")
a2.axhline(0, color=MUTED, lw=0.8)
a2.set_aspect("equal")
a2.set_xlim(fp.real - 1.3, fp.real + 1.3); a2.set_ylim(0, 2.35)
a2.set_xlabel("Re $\\tau$ (m)"); a2.set_ylabel("Im $\\tau$ (m)")
a2.set_title("(b) stable cavity round trips", fontsize=9, color=INK)
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=7, frameon=False)
a2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("../halfplane.pdf", bbox_inches="tight"); fig.savefig("../halfplane.png", dpi=200, bbox_inches="tight")
print("fixed point", fp, "(A+D)/2", (A + D) / 2)
