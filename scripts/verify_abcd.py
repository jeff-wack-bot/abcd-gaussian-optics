"""Numerical checks of the claims in the ABCD / complex-beam-parameter writeup.

Convention: E = u(x,z) exp(i(kz - wt)), paraxial eq 2ik u_z + u_xx = 0 (one transverse dim),
Gaussian u_q(x) = exp(i k x^2 / (2q)), physical beams have Im q < 0  (q = z - i zR at distance z
past the waist).  Free space and thin lenses are propagated by an FFT angular-spectrum method that
is independent of the ABCD formalism being tested.
"""
import numpy as np
from numpy.polynomial.hermite import hermval

lam = 1e-3                      # mm (1 micron)
k = 2 * np.pi / lam
N, L = 2**14, 24.0              # grid points, window width (mm)
x = (np.arange(N) - N // 2) * (L / N)
kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
dx = L / N

def free(u, d):   # exact solution of the paraxial equation over distance d
    return np.fft.ifft(np.fft.fft(u) * np.exp(-1j * d * kx**2 / (2 * k)))

def lens(u, f):
    return u * np.exp(-1j * k * x**2 / (2 * f))

def Mfree(d): return np.array([[1.0, d], [0.0, 1.0]])
def Mlens(f): return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])
def mob(M, q): return (M[0, 0] * q + M[0, 1]) / (M[1, 0] * q + M[1, 1])

def width(q):  # 1/q = 1/R + i lam/(pi w^2)
    return np.sqrt(lam / (np.pi * np.imag(1 / q)))

def hg(n, q):
    """Standard Hermite-Gauss form with real Hermite argument, normalised."""
    w = width(q)
    u = hermval(np.sqrt(2) * x / w, [0] * n + [1]) * np.exp(1j * k * x**2 / (2 * q))
    return u / np.sqrt(np.sum(abs(u)**2) * dx)

def inner(a, b): return np.sum(np.conj(a) * b) * dx

# ---- an arbitrary optical system: list of (kind, parameter)
system = [("d", 300.0), ("f", 250.0), ("d", 400.0), ("f", -150.0), ("d", 120.0),
          ("f", 80.0), ("d", 700.0), ("f", 300.0), ("d", 250.0)]
w0 = 0.5
zR = np.pi * w0**2 / lam
q0 = -1j * zR  # waist

print("== 1. ABCD law and Gouy phase vs. direct wave propagation ==")
for n in range(4):
    u = hg(n, q0)
    q, M, gouy = q0, np.eye(2), 0.0
    for kind, p in system:
        if kind == "d":
            u = free(u, p); E = Mfree(p)
            gouy += np.angle(1 + p / q)     # continuous along the segment (stays in upper half plane)
        else:
            u = lens(u, p); E = Mlens(p)
        q = mob(E, q); M = E @ M
    A, B, C, D = M.ravel()
    pred = hg(n, mob(M, q0)) * np.exp(-1j * (n + 0.5) * gouy)
    ov = inner(pred, u)
    print(f"n={n}: det M={np.linalg.det(M):.12f}  |<pred|num>|={abs(ov):.10f}  "
          f"phase err={np.angle(ov):+.2e} rad   "
          f"sum arg = {gouy:.4f}, principal arg(A+B/q0) = {np.angle(A + B / q0):.4f}")

print("\n== 2. Mode overlap = hyperbolic distance ==")
rng = np.random.default_rng(1)
for _ in range(4):
    qa = complex(rng.uniform(-500, 500), -rng.uniform(100, 1500))
    qb = complex(rng.uniform(-500, 500), -rng.uniform(100, 1500))
    ua, ub = hg(0, qa), hg(0, qb)
    num = abs(inner(ua, ub))**2
    cosh_d = 1 + abs(qa - qb)**2 / (2 * qa.imag * qb.imag)
    d = np.arccosh(cosh_d)
    print(f"|<a|b>|^2 = {num:.10f}   sech(d/2) = {1/np.cosh(d/2):.10f}   (d={d:.3f})")

print("\n== 3. Second moments: Sigma = (lambdabar/2|Im q|) Re(v v^H), v=(q,1); Sigma -> M Sigma M^T ==")
lb = 1 / k
def moments(u):
    p = np.abs(u)**2; p /= p.sum()
    U = np.fft.fft(u); P = np.abs(U)**2; P /= P.sum()
    xx = np.sum(p * x**2); th = kx / k
    tt = np.sum(P * th**2)
    du = np.gradient(u, dx)
    xt = np.real(np.sum(x * np.conj(u) * (-1j / k) * du) / np.sum(abs(u)**2))
    return np.array([[xx, xt], [xt, tt]])
q1 = complex(200.0, -600.0)
v = np.array([q1, 1.0])
S_pred = lb / (2 * abs(q1.imag)) * np.real(np.outer(v, v.conj()))
S_num = moments(hg(0, q1))
print("predicted\n", S_pred, "\nnumerical\n", S_num)
print("det Sigma / (lambdabar/2)^2 =", np.linalg.det(S_num) / (lb / 2)**2, " (=1: minimum uncertainty, M^2 = 1)")

print("\n== 4. Resonator: eigenvalue of the complex ray = round-trip Gouy factor ==")
R1, R2, Lc = 1000.0, 1500.0, 600.0     # two-mirror cavity, mirror = lens f = R/2
Mrt = Mlens(R1 / 2) @ Mfree(Lc) @ Mlens(R2 / 2) @ Mfree(Lc)
A, B, C, D = Mrt.ravel()
disc = (A + D)**2 - 4
qs = [((A - D) + s * np.sqrt(complex(disc))) / (2 * C) for s in (1, -1)]
qc = [q for q in qs if q.imag < 0][0]
lam_ev = C * qc + D
print(f"(A+D)/2 = {(A+D)/2:.4f}; fixed point q = {qc:.3f}; Cq+D = {lam_ev:.6f} (|.|={abs(lam_ev):.12f})")
th = np.arccos((A + D) / 2)
print(f"arccos((A+D)/2) = {th:.6f}, arg(Cq+D) = {np.angle(lam_ev):.6f}")
g1, g2 = 1 - Lc / R1, 1 - Lc / R2
print(f"textbook: round-trip Gouy (one transverse dim) = arccos(+-sqrt(g1 g2)) -> 2*arccos(sqrt(g1g2))={2*np.arccos(np.sqrt(g1*g2)):.6f}")
u = hg(0, qc)
w = free(u, Lc); w = lens(w, R2 / 2); w = free(w, Lc); w = lens(w, R1 / 2)
ov = inner(u, w)
print(f"numerical round trip: |<u|Uu>| = {abs(ov):.8f}, phase = {np.angle(ov):+.6f}; predicted -arg(Cq+D)/2 = {-np.angle(lam_ev)/2:+.6f}")

print("\n== 5. Metaplectic sign: a full period of a GRIN guide gives M = I but u -> -u (1D) ==")
# n(x) = n0 (1 - x^2/(2 h^2)) -> ray matrix rotation with period 2*pi*h ; split-step propagation
h = 200.0
period = 2 * np.pi * h
wg = np.sqrt(2 * h / k)            # matched width: zR = h
u0 = hg(0, -1j * h)
steps = 4000; dz = period / steps
u = u0.copy()
half = np.exp(-1j * (dz / 2) * kx**2 / (2 * k))
pot = np.exp(-1j * k * dz * x**2 / (2 * h**2))
for _ in range(steps):
    u = np.fft.ifft(np.fft.fft(u) * half); u = u * pot; u = np.fft.ifft(np.fft.fft(u) * half)
print(f"<u0|u(one period)> = {inner(u0, u):.6f}   (ray matrix of one period = identity)")
