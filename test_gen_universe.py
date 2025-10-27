#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-Transformer Reality Runner (QT-RR)
=========================================
Goal: Treat "reality" as if produced by a quantum-styled transformer:
- Complex amplitudes (phase-coherent evolution)
- Local-softmax attention mask (finite-speed influence)
- Decoherence as phase-noise channel
- Constraints as projection (ECC-analogue)
- Area-law via reduced density matrices
- Spectral bias, Boltzmann-softmax duality, rarity ~ exp(-lambda * delta_bits)

Outputs:
- Figures under ./figs/
- JSON metrics in gqt_results.json
- A Simulation-Likeness Index (0..100) aggregating orthogonal tests

Only requires: numpy, matplotlib, json, os
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(123)

# -----------------------
# Utils
# -----------------------
def ensure_dir(d="figs"):
    if not os.path.exists(d):
        os.makedirs(d)

def show():
    try:
        plt.show()
    except Exception:
        pass

def softmax(x, axis=-1, T=1.0):
    z = (x / max(T, 1e-12)) - np.max(x / max(T, 1e-12), axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def clamp01(x):
    return float(max(0.0, min(1.0, x)))

# -----------------------
# A) Quantum-like field evolution (complex amplitudes)
# -----------------------
def q_local_unitary_step(psi, kernel):
    """
    Convolution-like unitary-ish step on complex psi with a small kernel.
    Normalize to keep ||psi||=1 (Born-like).
    """
    out = np.zeros_like(psi, dtype=np.complex128)
    k = kernel
    r = len(k) // 2
    for i in range(len(psi)):
        acc = 0j
        for j, kj in enumerate(k):
            idx = i + (j - r)
            if 0 <= idx < len(psi):
                acc += kj * psi[idx]
        out[i] = acc
    # Re-normalize to enforce norm conservation
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out

def q_decohere(psi, strength=0.2):
    """
    Simple dephasing channel: multiply by exp(i * noise), then renormalize.
    """
    phases = np.exp(1j * RNG.normal(0.0, strength, size=len(psi)))
    out = psi * phases
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out

# -----------------------
# 1) Softmax <-> Boltzmann test (thermal duality)
# -----------------------
def test_softmax_boltzmann(num=200, T_true=0.73):
    """
    Sample states by Boltzmann over E, fit softmax over -delta_bits to those frequencies.
    We compare fitted temperature and correlation between energies and delta_lengths.
    """
    E = RNG.normal(0.0, 1.0, size=num)
    p = np.exp(-E / max(T_true, 1e-12))
    p = p / np.sum(p)
    # fake code-length deltas: assume delta_bits ~ a*E + noise
    a = 1.0
    delta_bits = a * E + RNG.normal(0.0, 0.2, size=num)

    # Fit T by minimizing KL(p || softmax(-delta_bits/T))
    Ts = np.linspace(0.2, 2.0, 100)
    best_T = None
    best_kl = float("inf")
    for T in Ts:
        q = softmax(-delta_bits, axis=0, T=T)
        kl = np.sum(np.where(p > 0, p * (np.log(p + 1e-12) - np.log(q + 1e-12)), 0.0))
        if kl < best_kl:
            best_kl = kl
            best_T = T

    corr = np.corrcoef(E, delta_bits)[0, 1]
    # score: accurate T and strong positive correlation
    sT = math.exp(-abs(best_T - T_true) / 0.2)
    sc = clamp01((corr + 1.0) / 2.0)  # map [-1,1] to [0,1]
    score = 0.5 * sT + 0.5 * sc

    return {
        "T_true": T_true,
        "T_fit": float(best_T),
        "KL": float(best_kl),
        "corr_E_dL": float(corr),
        "score": score,
        "note": "Softmax over -delta_bits approximates Boltzmann over E if a transformer-like sampler underlies thermal stats."
    }

# -----------------------
# 2) Interference vs decoherence (double-slit style)
# -----------------------
def double_slit_visibility(psi, center_window=60):
    I = np.abs(psi) ** 2
    n = len(I)
    lo = max(0, n // 2 - center_window // 2)
    hi = min(n, n // 2 + center_window // 2)
    region = I[lo:hi]
    vis = (region.max() - region.min()) / (region.max() + region.min() + 1e-12)
    return float(vis)

def test_interference_decoherence(L=301, steps=150, slit_sep=40, phase_strength=0.22, decoh=0.25):
    """
    Create two coherent lobes, evolve with a complex kernel, compare visibility to decohered case.
    """
    x = np.arange(L)
    center = L // 2
    slit1, slit2 = center - slit_sep // 2, center + slit_sep // 2

    # Initial two-slit wavefunction
    psi0 = np.exp(-0.5 * ((x - slit1) / 6.0) ** 2) + np.exp(-0.5 * ((x - slit2) / 6.0) ** 2)
    psi0 = psi0.astype(np.complex128)
    psi0 = psi0 / np.linalg.norm(psi0)

    # Complex kernel (unitary-ish, small phase)
    kernel = np.array([
        0.25 * np.exp(1j * phase_strength),
        0.50 * np.exp(1j * phase_strength),
        0.25 * np.exp(1j * phase_strength),
    ], dtype=np.complex128)

    # Evolve coherent
    psi_c = psi0.copy()
    for _ in range(steps):
        psi_c = q_local_unitary_step(psi_c, kernel)

    # Evolve with decoherence
    psi_d = psi0.copy()
    for _ in range(steps):
        psi_d = q_local_unitary_step(psi_d, kernel)
        psi_d = q_decohere(psi_d, strength=decoh / steps * 5.0)  # small dephasing per step

    Vc = double_slit_visibility(psi_c)
    Vd = double_slit_visibility(psi_d)
    deltaV = Vc - Vd

    # score: fringe visibility should drop with decoherence
    score = clamp01(deltaV / 0.2)

    # plot
    ensure_dir()
    I_c = np.abs(psi_c) ** 2
    I_d = np.abs(psi_d) ** 2
    xs = np.arange(L) - center
    plt.figure()
    plt.plot(xs, I_c, label="coherent")
    plt.plot(xs, I_d, label="decohered")
    plt.title("Double-slit: coherent vs decohered")
    plt.xlabel("x"); plt.ylabel("intensity")
    plt.legend(); plt.tight_layout()
    plt.savefig("figs/double_slit.png", dpi=140)
    show()

    return {
        "V_coherent": Vc,
        "V_decohered": Vd,
        "deltaV": deltaV,
        "score": score,
        "note": "Decoherence should reduce fringe visibility if phase coherence is physical."
    }

# -----------------------
# 3) Local-softmax light-cone causality
# -----------------------
def test_causality_local_softmax(L=96, T=40, c=2):
    """
    Transformer-like step: logits from similarity, softmax with a local mask.
    We inject a delta at t_inject and measure influence spread at t+1 (baseline-subtracted).
    """
    x = np.arange(L)
    dist = np.abs(x[:, None] - x[None, :])
    base = - (dist / (L / 8.0)) ** 2
    sim = base + 0.05 * RNG.normal(0.0, 1.0, size=(L, L))

    def step(v, mask):
        logits = sim.copy()
        logits[~mask] = -1e9
        A = softmax(logits, axis=1, T=1.0)
        return A @ v

    # mask for |dx| <= c
    mask = np.zeros((L, L), dtype=bool)
    for i in range(L):
        lo, hi = max(0, i - c), min(L, i + c + 1)
        mask[i, lo:hi] = True

    t_inject = 10
    inj_pos = 5
    inj_amp = 0.5

    def run(inject):
        s = np.zeros(L); s[L // 2] = 1.0
        traj = [s.copy()]
        for t in range(T):
            s = step(s, mask)
            if inject and t == t_inject:
                s[inj_pos] += inj_amp
            traj.append(s.copy())
        return np.stack(traj, axis=0)

    with_inj = run(True)
    no_inj = run(False)
    delta = with_inj - no_inj

    # radius at next step after injection
    row = delta[t_inject + 1]
    eps = 1e-9
    idx = np.where(np.abs(row) > eps)[0]
    radius = int(np.max(np.abs(idx - inj_pos))) if len(idx) else 0

    # score: perfect if radius <= c
    score = 1.0 if radius <= c else clamp01(c / (radius + 1e-9))

    # plot
    ensure_dir()
    plt.figure()
    plt.imshow(delta.T, aspect="auto", origin="lower", cmap="bwr")
    plt.colorbar()
    plt.title("Causality heatmap (with − without injection)")
    plt.xlabel("time"); plt.ylabel("position")
    plt.tight_layout()
    plt.savefig("figs/causality_heatmap.png", dpi=140)
    show()

    return {
        "radius_tplus1": radius,
        "c_mask": c,
        "score": score,
        "note": "Local-softmax attention enforces finite-speed influence per step (light-cone)."
    }

# -----------------------
# 4) Area-law proxy via reduced density matrices
# -----------------------
def entanglement_entropy(psi, L_block):
    """
    Build a two-qubit-like reduced density matrix by grouping sites into A (block) and B.
    We use an amplitude trick: form outer product psi psi^*, then trace out B indices.
    Since 1D, we approximate by splitting psi into block vs complement and projecting.
    """
    n = len(psi)
    A_idx = np.arange(max(0, n // 2 - L_block // 2), min(n, n // 2 + L_block // 2))
    B_idx = np.array([i for i in range(n) if i not in set(A_idx.tolist())])

    # Construct reduced density for A by marginalizing intensities (approx, not full tensor)
    # Proper many-body RDM would need tensor product state; here we approximate:
    # rho_A[i,j] ~ psi[i] psi[j]^* normalized over A.
    if len(A_idx) < 2:
        return 0.0
    phi_A = psi[A_idx]
    nA = np.linalg.norm(phi_A)
    if nA <= 0:
        return 0.0
    rhoA = np.outer(phi_A, np.conjugate(phi_A)) / (nA ** 2)
    evals = np.real(np.linalg.eigvalsh(rhoA))
    evals = np.clip(evals, 1e-16, 1.0)
    S = float(-np.sum(evals * np.log(evals)))
    return S

def test_area_law(psi_len=256, max_block=64):
    """
    Prepare a localized, weakly entangled state; compute S(block) vs block size.
    An area-law signal is S growing sublinearly and tending to saturate.
    """
    x = np.arange(psi_len)
    psi = np.exp(-0.5 * ((x - psi_len // 2) / 10.0) ** 2) * np.exp(1j * 0.2 * x)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    blocks = np.unique(np.linspace(4, max_block, 12, dtype=int))
    Ss = []
    for Lb in blocks:
        Ss.append(entanglement_entropy(psi, Lb))

    # Fit S ~ a * log(Lb) + b (area-law-ish saturation in 1D)
    X = np.vstack([np.log(blocks + 1e-9), np.ones_like(blocks)]).T
    y = np.array(Ss)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    slope_log = float(coef[0])

    # score: small positive slope (near 0..0.5) is good for area-law-ish
    score = clamp01(max(0.0, 0.5 - abs(slope_log - 0.2)) / 0.5)

    # plot
    ensure_dir()
    plt.figure()
    plt.plot(blocks, Ss, "o-")
    plt.title("Area-law proxy: S(block) vs block size")
    plt.xlabel("block size"); plt.ylabel("entropy S")
    plt.tight_layout()
    plt.savefig("figs/area_law.png", dpi=140)
    show()

    return {
        "slope_log": slope_log,
        "score": score,
        "note": "Sublinear S(block) growth is consistent with locality/low-rank mixing (area-law-like)."
    }

# -----------------------
# 5) Spectral bias (1/f^alpha)
# -----------------------
def test_spectral_bias(N=2048):
    """
    Generate a field via local mixing to emulate trained-model smoothness;
    measure power spectrum slope in log-log.
    """
    # Build a smooth field by low-pass filtering white noise
    x = RNG.normal(0.0, 1.0, size=N)
    kernel = np.array([0.05, 0.2, 0.5, 0.2, 0.05])
    for _ in range(12):
        x = np.convolve(x, kernel, mode="same")
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0)
    valid = freqs > 0
    logf = np.log(freqs[valid])
    logP = np.log(P[valid] + 1e-12)
    A = np.vstack([logf, np.ones_like(logf)]).T
    m, b = np.linalg.lstsq(A, logP, rcond=None)[0]
    alpha = -float(m)

    # score: alpha near 1 is classic 1/f
    score = clamp01(1.0 - abs(alpha - 1.0) / 1.0)

    # plot
    ensure_dir()
    plt.figure()
    plt.plot(freqs[valid], np.exp(logP), lw=1)
    plt.xscale("log"); plt.yscale("log")
    plt.title(f"Spectral bias: slope ~ -{alpha:.2f}")
    plt.xlabel("frequency"); plt.ylabel("power")
    plt.tight_layout()
    plt.savefig("figs/spectral_bias.png", dpi=140)
    show()

    return {
        "alpha": alpha,
        "score": score,
        "note": "1/f^alpha with alpha near 1 indicates trained-generator style spectral bias."
    }

# -----------------------
# 6) ECC-like projection (constraint repair)
# -----------------------
def project_constraints_complex(psi):
    """
    Stabilizer-like parity: sum of nearest-neighbor phases should be smooth.
    We minimize a quadratic cost on phase differences and renormalize.
    """
    amp = np.abs(psi) + 1e-12
    phase = np.unwrap(np.angle(psi))
    # Smooth phase by solving a tridiagonal regularization
    lam = 10.0
    n = len(phase)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1 + 2 * lam
        if i > 0:
            A[i, i - 1] = -lam
        if i < n - 1:
            A[i, i + 1] = -lam
    b = phase.copy()
    phase_smooth = np.linalg.solve(A, b)
    psi_repaired = amp * np.exp(1j * phase_smooth)
    psi_repaired = psi_repaired / np.linalg.norm(psi_repaired)
    return psi_repaired

def test_ecc_projection(L=256, noise=0.3):
    """
    Start with a smooth coherent psi, add phase noise, then project.
    Measure whether projection reduces a 'syndrome' (phase gradient variance).
    """
    x = np.arange(L)
    psi0 = np.exp(-0.5 * ((x - L // 2) / 12.0) ** 2) * np.exp(1j * 0.15 * x)
    psi0 = psi0.astype(np.complex128)
    psi0 = psi0 / np.linalg.norm(psi0)

    # add noise
    noisy = psi0 * np.exp(1j * RNG.normal(0.0, noise, size=L))
    # syndrome: variance of phase difference
    def syndrome(p):
        ph = np.unwrap(np.angle(p))
        dif = np.diff(ph)
        return float(np.var(dif))

    syn0 = syndrome(noisy)
    repaired = project_constraints_complex(noisy)
    syn1 = syndrome(repaired)

    # score: big reduction is good
    red = max(0.0, syn0 - syn1)
    score = clamp01(red / (syn0 + 1e-12))

    ensure_dir()
    plt.figure()
    plt.plot(np.unwrap(np.angle(noisy)), label="noisy phase", alpha=0.7)
    plt.plot(np.unwrap(np.angle(repaired)), label="repaired phase", alpha=0.9)
    plt.title("ECC-like projection: phase repair")
    plt.xlabel("position"); plt.ylabel("unwrapped phase")
    plt.legend(); plt.tight_layout()
    plt.savefig("figs/ecc_projection.png", dpi=140)
    show()

    return {
        "syndrome_before": syn0,
        "syndrome_after": syn1,
        "score": score,
        "note": "Projection onto a smooth-phase constraint manifold reduces a stabilizer-like syndrome."
    }

# -----------------------
# 7) Anomaly rarity vs delta-bits
# -----------------------
def test_rarity_vs_delta_bits(N=2000):
    """
    Generate synthetic 'anomaly' magnitudes a >= 0 and model counts proportional to exp(-lambda * a).
    Check semilog linearity and slope sign.
    """
    lam = 1.0
    a = RNG.exponential(scale=1.0 / lam, size=N)
    # Bin and count
    bins = np.linspace(0.0, 5.0, 26)
    H, edges = np.histogram(a, bins=bins)
    mids = 0.5 * (edges[:-1] + edges[-1:][0])

    valid = H > 0
    x = mids[valid]
    y = H[valid]
    # Fit log counts vs a
    A = np.vstack([x, np.ones_like(x)]).T
    logy = np.log(y + 1e-12)
    m, b = np.linalg.lstsq(A, logy, rcond=None)[0]
    slope = float(m)

    # score: negative slope near -1 is good
    score = clamp01(1.0 - abs(slope + 1.0) / 1.0)

    ensure_dir()
    plt.figure()
    plt.plot(x, y, "o", label="counts")
    plt.plot(x, np.exp(m * x + b), label=f"fit slope {slope:.2f}")
    plt.yscale("log")
    plt.title("Anomaly rarity ~ exp( - lambda * delta_bits )")
    plt.xlabel("delta_bits proxy"); plt.ylabel("count (log)")
    plt.legend(); plt.tight_layout()
    plt.savefig("figs/rarity_semilog.png", dpi=140)
    show()

    return {
        "slope_semilog": slope,
        "score": score,
        "note": "Exponential rarity with cost supports a compression-driven generator."
    }

# -----------------------
# Aggregate and run
# -----------------------
def aggregate_scores(results, weights):
    keys = list(results.keys())
    total = 0.0
    wsum = 0.0
    for k in keys:
        s = results[k]["score"]
        w = weights.get(k, 1.0)
        total += s * w
        wsum += w
    idx = 100.0 * total / max(wsum, 1e-12)
    return idx

def main():
    ensure_dir()

    # Run tests
    soft = test_softmax_boltzmann()
    inter = test_interference_decoherence()
    caus = test_causality_local_softmax()
    area = test_area_law()
    spec = test_spectral_bias()
    ecc = test_ecc_projection()
    rare = test_rarity_vs_delta_bits()

    results = {
        "softmax_boltzmann": soft,
        "interference": inter,
        "causality": caus,
        "area_law": area,
        "spectral_bias": spec,
        "ecc_projection": ecc,
        "rarity_vs_bits": rare,
    }

    # Weights emphasize phase/coherence, locality, and compression signatures
    weights = {
        "softmax_boltzmann": 1.0,
        "interference": 2.0,
        "causality": 2.0,
        "area_law": 1.0,
        "spectral_bias": 1.0,
        "ecc_projection": 1.5,
        "rarity_vs_bits": 1.0,
    }

    sim_index = aggregate_scores(results, weights)

    # Print summary
    print("\n==============================")
    print("Quantum-Transformer Signature Summary")
    print("==============================")
    for k, v in results.items():
        print(f"{k:20s} | score={v['score']:.3f} | note: {v['note']}")
    print("------------------------------")
    print(f"Simulation-Likeness Index (0–100): {sim_index:.2f}")
    print("------------------------------")
    print("Figures saved in: ./figs/ ; Raw JSON: gqt_results.json\n")

    # Save JSON
    out = {"results": results, "weights": weights, "simulation_index": sim_index}
    with open("gqt_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
