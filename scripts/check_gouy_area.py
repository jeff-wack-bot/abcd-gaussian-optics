"""Check: Gouy phase difference = -1/4 x hyperbolic area of the geodesic quadrilateral with vertices
0 (point source / delta), tau1, infinity (plane wave), tau2   [Simon & Mukunda 1993; Mukunda, Chaturvedi, Simon 2014].
Work in the upper half-plane tau = conj(q) = z + i zR.  Ideal vertices have angle 0, so area = 2*pi - (a1 + a2),
where a_j is the interior angle at tau_j between the geodesic to 0 and the geodesic (vertical line) to infinity."""
import numpy as np

def tangent_to_zero(t):
    # geodesic through t and 0: circle centred on real axis at c with |t-c| = |c|  ->  c = |t|^2 / (2 Re t)
    c = abs(t)**2 / (2 * t.real)
    r = t - c
    tan = 1j * r                          # tangent direction of the circle at t
    # orient it towards 0: pick sign so that moving along it decreases |t|
    if abs(t + 1e-6 * tan / abs(tan)) > abs(t):
        tan = -tan
    return tan / abs(tan)

def interior_angle(t):
    up = 1j                               # towards infinity
    return abs(np.angle(tangent_to_zero(t) / up))

zR = 1.0
for z1, z2 in [(-0.7, 1.9), (-3.0, -0.2), (0.3, 5.0)]:
    t1, t2 = z1 + 1j * zR, z2 + 1j * zR
    area = 2 * np.pi - interior_angle(t1) - interior_angle(t2)
    gouy = 0.5 * (np.arctan(z2 / zR) - np.arctan(z1 / zR))   # magnitude of 1D Gouy phase change
    print(f"z1={z1:+.1f} z2={z2:+.1f}:  area/4 = {area/4:.6f}   |Delta phi_G| (1D) = {gouy:.6f}")

print("signed version: area = difference of triangles (0, tau, inf), each of area pi - angle, signed by side of waist")
def tri(t):  # signed area of geodesic triangle (0, t, inf): positive if t right of waist
    return np.sign(t.real) * (np.pi - interior_angle(t))
for z1, z2 in [(-0.7, 1.9), (-3.0, -0.2), (0.3, 5.0), (-10.0, 10.0)]:
    t1, t2 = z1 + 1j * zR, z2 + 1j * zR
    area = tri(t2) - tri(t1)
    gouy = 0.5 * (np.arctan(z2 / zR) - np.arctan(z1 / zR))
    print(f"z1={z1:+.1f} z2={z2:+.1f}:  signed area/4 = {area/4:.6f}   Delta(1/2 arctan) = {gouy:.6f}")
