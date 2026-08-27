"""
potential.py
============
CG potential parameter extraction via Boltzmann inversion.

Pipeline:
  1. Read fin.lammpstrj (trajectory) frame by frame
  2. For each frame: compute bead COM positions (PBC-aware) using bead_map_def
  3. Accumulate bond/angle distributions and RDF
  4. Boltzmann inversion -> PMF
  5. Fit harmonic (bond, angle) and LJ 12-6 (nonbonded) to PMF
  6. Return {r0, k_bond, theta0, k_angle, epsilon, sigma}
"""

from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

KB_KCAL = 0.0019872041   # kcal / (mol·K)


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY READER
# ─────────────────────────────────────────────────────────────────────────────

def iter_frames(trj_path: str):
    """
    Yield (timestep, box_np, coords_dict) for each frame in a LAMMPS dump file.
    box_np : np.array shape (3,2) [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    coords : {atom_id: (x, y, z)}  (xu/yu/zu or x/y/z)
    """
    with open(trj_path, encoding='utf-8', errors='replace') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(f.readline().strip())

            f.readline()                          # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())

            f.readline()                          # ITEM: BOX BOUNDS
            box = []
            for _ in range(3):
                lo, hi = f.readline().split()[:2]
                box.append([float(lo), float(hi)])
            box = np.array(box)

            col_line = f.readline()               # ITEM: ATOMS id ...
            cols = col_line.strip().split()[2:]
            ci   = {c: i for i, c in enumerate(cols)}

            if "xu" in ci:
                xc, yc, zc = "xu", "yu", "zu"
            else:
                xc, yc, zc = "x", "y", "z"

            coords = {}
            for _ in range(natoms):
                p = f.readline().split()
                aid = int(p[ci["id"]])
                coords[aid] = (float(p[ci[xc]]),
                               float(p[ci[yc]]),
                               float(p[ci[zc]]))
            yield timestep, box, coords


# ─────────────────────────────────────────────────────────────────────────────
# BEAD POSITION FROM COORDS  (PBC-aware COM)
# ─────────────────────────────────────────────────────────────────────────────

def _mic(dr: np.ndarray, L: np.ndarray) -> np.ndarray:
    return dr - L * np.round(dr / L)


def _com_pbc(positions: np.ndarray, masses: np.ndarray, L: np.ndarray) -> np.ndarray:
    ref = positions[0].copy()
    uw  = positions.copy()
    for i in range(1, len(positions)):
        dr = positions[i] - ref
        uw[i] = ref + _mic(dr, L)
    return np.average(uw, axis=0, weights=masses)


def bead_positions_from_frame(
    bead_list:   list[tuple],          # [(mol_id, [atom_ids]), ...]
    atom_masses: dict[int, float],     # atom_id -> mass
    coords:      dict[int, tuple],     # atom_id -> (x,y,z)
    box:         np.ndarray,           # (3,2)
) -> np.ndarray:
    """
    Compute (N_beads, 3) array of COM positions for one frame.
    bead_list: [(mol_id, [atom_id, ...]), ...] — one entry per bead
    """
    L  = box[:, 1] - box[:, 0]
    lo = box[:, 0]
    out = np.zeros((len(bead_list), 3))
    for k, (_mid, atom_ids) in enumerate(bead_list):
        pos = np.array([coords[a] for a in atom_ids], dtype=float)
        ms  = np.array([atom_masses[a] for a in atom_ids], dtype=float)
        # wrap to [0, L)
        pos = (pos - lo) % L
        com = _com_pbc(pos, ms, L)
        out[k] = com % L
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILD BEAD LIST FROM DATA FILE + bead_map_def
# ─────────────────────────────────────────────────────────────────────────────

def build_bead_list(
    aa_data_path: str,
    poly_def:     dict,
    bead_map_def: dict,
    n_chains:     int,
    n_monomers:   int,
) -> tuple[list, dict, list]:
    """
    Returns:
      bead_list      : [(mol_id, [atom_ids]), ...]
      atom_masses    : {atom_id: mass}
      mol_bead_ids   : [[bead_idx, ...], ...]  per molecule (for bond/angle pairs)
    """
    from mapper import read_aa_data, build_mol_adj, linear_chain_order
    from polymer import normalize_poly_def, compute_monomer_offsets

    atoms, bonds, box, masses_map = read_aa_data(aa_data_path)
    mol_atoms, adj = build_mol_adj(atoms, bonds)

    poly_def_n    = normalize_poly_def(poly_def, n_monomers)
    sequence      = poly_def_n["sequence"]
    monomer_types = poly_def_n["monomer_types"]
    bead_types    = bead_map_def["bead_types"]
    has_mono_type = any("monomer_type" in bt for bt in bead_types)
    max_mpb       = max((bt.get("monomers_per_bead") or bt.get("repeating_monomers",1) or 1) for bt in bead_types)

    bead_list:    list[tuple]      = []
    atom_masses:  dict[int, float] = {aid: masses_map.get(a.atype, 12.011) for aid, a in atoms.items()}
    mol_bead_ids: list[list[int]]  = []

    for mid in sorted(mol_atoms.keys()):
        atom_order    = linear_chain_order(mol_atoms[mid], adj)
        n_chain       = len(atom_order)
        offsets       = compute_monomer_offsets(sequence, monomer_types)
        monomer_groups = []
        for mi, lbl in enumerate(sequence):
            s, e = offsets[mi], offsets[mi+1]
            if e <= n_chain:
                monomer_groups.append((lbl, atom_order[s:e]))

        mol_bead_start = len(bead_list)
        mono_idx = 0
        while mono_idx < len(monomer_groups):
            cur_lbl    = monomer_groups[mono_idx][0]
            applicable = ([bt for bt in bead_types if bt.get("monomer_type") == cur_lbl]
                         or [bt for bt in bead_types if "monomer_type" not in bt]) if has_mono_type else bead_types
            for bt in applicable:
                lids = bt.get("atom_local_ids") or bt.get("local_ids_in_monomer", [])
                mpb  = bt.get("monomers_per_bead") or bt.get("repeating_monomers",1) or 1
                bead_atom_ids = []
                for m_off in range(mpb):
                    mi = mono_idx + m_off
                    if mi >= len(monomer_groups): break
                    _, mg = monomer_groups[mi]
                    for lid in lids:
                        idx = lid - 1
                        if 0 <= idx < len(mg):
                            bead_atom_ids.append(mg[idx])
                if bead_atom_ids:
                    bead_list.append((mid, bead_atom_ids))
            mono_idx += max_mpb

        mol_bead_ids.append(list(range(mol_bead_start, len(bead_list))))

    return bead_list, atom_masses, mol_bead_ids


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION ACCUMULATORS
# ─────────────────────────────────────────────────────────────────────────────

def _bond_angle_pairs(mol_bead_ids: list[list[int]]):
    """Return bond_pairs and angle_triplets as numpy arrays (bead indices)."""
    bonds, angles = [], []
    for bids in mol_bead_ids:
        for i in range(len(bids) - 1):
            bonds.append([bids[i], bids[i+1]])
        for i in range(len(bids) - 2):
            angles.append([bids[i], bids[i+1], bids[i+2]])
    return (np.array(bonds,  dtype=int) if bonds  else np.empty((0,2), int),
            np.array(angles, dtype=int) if angles else np.empty((0,3), int))


def _compute_bonds(pos: np.ndarray, pairs: np.ndarray, L: np.ndarray) -> np.ndarray:
    dr = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    dr = _mic(dr, L)
    return np.linalg.norm(dr, axis=1)


def _compute_angles(pos: np.ndarray, trips: np.ndarray, L: np.ndarray) -> np.ndarray:
    v1 = _mic(pos[trips[:, 0]] - pos[trips[:, 1]], L)
    v2 = _mic(pos[trips[:, 2]] - pos[trips[:, 1]], L)
    cos = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1) + 1e-30)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _compute_rdf(pos: np.ndarray, L: np.ndarray,
                 r_max: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, r_max, n_bins + 1)
    n = len(pos)
    hist = np.zeros(n_bins, dtype=np.int64)
    # vectorised pairwise (cap at 2000 beads to avoid memory issues)
    if n <= 2000:
        for i in range(n):
            dr = _mic(pos[i+1:] - pos[i], L)
            r  = np.linalg.norm(dr, axis=1)
            mask = r < r_max
            h, _ = np.histogram(r[mask], bins=edges)
            hist += h
    return hist, edges


# ─────────────────────────────────────────────────────────────────────────────
# BOLTZMANN INVERSION
# ─────────────────────────────────────────────────────────────────────────────

def boltzmann_inversion_bond(centers: np.ndarray, pdf: np.ndarray,
                              kBT: float) -> np.ndarray:
    """U(r) = -kBT * ln[ P(r) ]

    The r² (spherical Jacobian) correction is intentionally omitted for CG bonds.
    That correction is appropriate for non-bonded RDFs where the reference state
    is the ideal gas in 3D space.  For a CG bond — which represents a covalent
    or quasi-covalent connectivity between two specific beads — the natural
    reference is the 1D harmonic potential, and the raw histogram P(r) already
    samples the bond distribution correctly without a geometric weight.
    Applying 1/r² would artificially shift r0 ~0.3–0.5 Å toward shorter distances.
    """
    safe = np.where(pdf > 0, pdf, np.nan)
    pmf  = -kBT * np.log(safe)
    return pmf


def boltzmann_inversion_angle(centers: np.ndarray, pdf: np.ndarray,
                               kBT: float) -> np.ndarray:
    """U(θ) = -kBT * ln[ P(θ) / sin(θ) ]"""
    sin_t = np.abs(np.sin(np.radians(centers))) + 1e-30
    safe  = np.where(pdf > 0, pdf, np.nan)
    pmf   = -kBT * np.log(safe / sin_t)
    return pmf


def boltzmann_inversion_rdf(centers: np.ndarray, g_r: np.ndarray,
                             kBT: float) -> np.ndarray:
    """U(r) = -kBT * ln[ g(r) ]"""
    safe = np.where(g_r > 0, g_r, np.nan)
    return -kBT * np.log(safe)


# ─────────────────────────────────────────────────────────────────────────────
# FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_harmonic_bond(centers: np.ndarray, pmf: np.ndarray,
                      kBT: float = 0.5961) -> tuple[float, float]:
    """
    Extract harmonic parameters from the PMF curvature near its minimum.
    Returns (K [kcal/mol/Å²], r0 [Å])  where K is the LAMMPS coefficient
    in E = K*(r - r0)^2  (no 1/2 prefactor).

    Physical basis
    --------------
    For a harmonic CG bond  U(r) = (1/2) k (r-r0)^2  the LAMMPS parameter
    is K = k/2.  Fitting U_PMF ≈ a*(r-r0)^2 near the minimum gives a = K
    directly, because  U_PMF = U_harmonic + const  near x0.

    Window strategy
    ---------------
    Only points within 1.5 kBT of the PMF minimum are used.  The CG bond PMF
    is typically asymmetric: a gradual left flank and a steep hard-core wall
    on the right.  The old wide window (PMF < 3 kcal/mol) included the entire
    wall, biasing the parabola rightward and making K artificially small (~9×).
    Restricting to ≤1.5 kBT keeps only the near-equilibrium region where the
    harmonic approximation is valid.
    """
    from scipy.optimize import curve_fit

    valid = np.isfinite(pmf)
    if valid.sum() < 4:
        if valid.any():
            return 0.0, float(centers[valid][np.argmin(pmf[valid])])
        return 0.0, 0.0

    pmf_s   = pmf - np.nanmin(pmf)
    r0_init = float(centers[valid][np.argmin(pmf[valid])])

    # Narrow window: thermal-fluctuation region only
    mask = valid & (pmf_s <= 1.5 * kBT)
    if mask.sum() < 4:
        mask = valid & (pmf_s <= 3.0 * kBT)
    if mask.sum() < 4:
        mask = valid

    x, y = centers[mask], pmf_s[mask]

    def _harmonic(r, K, r0):
        return K * (r - r0)**2

    try:
        popt, _ = curve_fit(_harmonic, x, y,
                            p0=[kBT / (r0_init * 0.01 + 0.01)**2, r0_init],
                            bounds=([0.0, 0.5 * r0_init],
                                    [1e4,  2.0 * r0_init + 1.0]),
                            maxfev=10000)
        K, r0 = float(popt[0]), float(popt[1])
    except Exception:
        r0    = float(x[np.argmin(y)])
        K_arr = y / ((x - r0)**2 + 1e-30)
        K     = float(np.median(K_arr[np.isfinite(K_arr)]))

    return max(K, 0.0), r0


def fit_harmonic_angle(centers: np.ndarray, pmf: np.ndarray,
                       kBT: float = 0.5961) -> tuple[float, float]:
    """
    Extract harmonic angle parameters from the PMF curvature near its minimum.
    Returns (K [kcal/mol/rad²], θ0 [deg])  where K is the LAMMPS coefficient
    in E = K*(θ - θ0)^2  (no 1/2 prefactor).

    The sin(θ) Jacobian correction is already applied in
    boltzmann_inversion_angle, so the PMF passed here already reflects the
    true angular free energy.

    Robust strategy
    ---------------
    θ0 must coincide with the most-probable angle (the PMF minimum / PDF peak).
    The previous complement-space fit could latch onto noisy boundary bins and
    return θ0≈180° (or 0°) even when the real peak was ~140°, producing a rigid
    rod that is longer than the simulation box and blows the CG density up.

    Here we (1) locate the minimum on a smoothed PMF restricted to bins with
    real data, (2) fit a parabola in a narrow ±window around it, and
    (3) sanity-check θ0 against the smoothed minimum, falling back to it if the
    fit wanders. Finally, near-180° equilibria are clamped away from a perfectly
    straight, stiff angle so chains can still fold.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit

    GUARD   = 2.0     # degrees — exclude noisy boundary bins
    K_MIN   = 0.5     # kcal/mol/rad²
    K_MAX   = 50.0    # kcal/mol/rad² — cap; CG angles are soft, not rigid
    # Above this θ0 a harmonic well is one-sided (θ can't exceed 180°); pull the
    # equilibrium angle back and keep K modest so the chain is not a rigid rod.
    STRAIGHT_THR = 170.0
    STRAIGHT_CAP = 165.0   # θ0 assigned when the measured minimum is ~straight
    STRAIGHT_K   = 2.0     # soft constant for near-straight CG angles

    valid = np.isfinite(pmf) & (centers >= GUARD) & (centers <= 180.0 - GUARD)
    # Empty histogram bins (zero pdf) are sometimes stored as pmf==0 instead of
    # NaN; these are NOT real minima. Exclude exact-zero pmf bins unless every
    # bin is zero (degenerate). This prevents θ0 latching onto an empty
    # low-angle region when the true minimum is, e.g., ~140°.
    nonzero_pmf = valid & (pmf != 0.0)
    if nonzero_pmf.sum() >= 4:
        valid = nonzero_pmf
    if valid.sum() < 4:
        anyv = np.isfinite(pmf)
        if anyv.any():
            return K_MIN, float(centers[anyv][np.argmin(pmf[anyv])])
        return K_MIN, 120.0

    # Smooth lightly over the valid region to find a reliable minimum. Invalid /
    # empty bins are filled with a high value so the smoothed minimum cannot
    # fall there. Light smoothing (sigma=1) avoids dragging the minimum away
    # from the true PDF peak in broad, asymmetric distributions.
    fill = np.nanmax(pmf[valid]) if valid.any() else 0.0
    pmf_work   = np.where(valid, pmf, fill)
    pmf_smooth = gaussian_filter1d(pmf_work, sigma=1, mode='nearest')
    pmf_smooth = pmf_smooth - np.nanmin(pmf_smooth[valid])

    th0_min = float(centers[valid][np.argmin(pmf_smooth[valid])])

    # The PMF minimum can sit at a high angle (~170°) purely because the sin(θ)
    # Jacobian in U=-kT ln(P/sinθ) divides by a small sin(θ) there. The physically
    # meaningful equilibrium angle is the most-probable angle, i.e. the peak of
    # the raw probability P(θ) ∝ sin(θ)·exp(-PMF/kT). Reconstruct P and use its
    # peak as θ0; this never yields a straight rigid rod for a chain whose angles
    # actually populate ~140°.
    P_recon = np.where(valid, np.sin(np.radians(centers)) * np.exp(-pmf_smooth / kBT), 0.0)
    P_recon = np.where(np.isfinite(P_recon), P_recon, 0.0)
    if P_recon.sum() > 0:
        th0_peak = float(centers[np.argmax(P_recon)])
    else:
        th0_peak = th0_min

    # Detect a FLAT / soft angle PMF (freely-bending chain). When the well depth
    # is small the harmonic θ0 is ill-defined; use the probability-weighted mean
    # angle and a soft K so the chain can fold rather than lock straight.
    span = float(np.nanmax(pmf_smooth[valid]) - np.nanmin(pmf_smooth[valid]))
    if span < 2.0 * kBT:
        w = P_recon
        th0_mean = float(np.sum(centers * w) / (np.sum(w) + 1e-30))
        th0_deg  = min(th0_mean, STRAIGHT_CAP)
        return float(min(max(STRAIGHT_K, K_MIN), K_MAX)), th0_deg

    # Otherwise anchor θ0 at the reconstructed-probability peak.
    th0_min = th0_peak

    # Narrow window around the smoothed minimum (thermal-fluctuation region)
    def _mask(thr):
        return valid & (pmf_smooth <= thr * kBT)
    m = _mask(1.5)
    if m.sum() < 4: m = _mask(3.0)
    if m.sum() < 4: m = valid

    x_rad = np.radians(centers[m])
    y     = pmf_smooth[m] - pmf_smooth[m].min()
    x0_rad = np.radians(th0_min)

    def _h(x, K, x0): return K * (x - x0)**2
    # Fit with θ0 FIXED at the data minimum; optimise only the curvature K.
    # In broad, asymmetric CG angle distributions a free θ0 drifts toward the
    # long tail; the most-probable angle (PDF peak = PMF minimum) is the correct
    # equilibrium value, so we anchor θ0 there and let the parabola give K.
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            lambda x, K: _h(x, K, x0_rad), x_rad, y,
            p0=[2.0], bounds=([0.0], [K_MAX]), maxfev=10000)
        K = float(popt[0])
    except Exception:
        K_arr = y / ((x_rad - x0_rad)**2 + 1e-30)
        K = float(np.median(K_arr[np.isfinite(K_arr)]))
    th0_deg = th0_min

    # Near-straight equilibrium → avoid a stiff, rigid-rod angle that can exceed
    # the box length. Pull θ0 back from 180° and use a soft constant so the chain
    # can fold during NPT.
    if th0_deg >= STRAIGHT_THR:
        th0_deg = STRAIGHT_CAP
        K       = min(K, STRAIGHT_K)

    if not (0.0 <= th0_deg <= 180.0):
        th0_deg = th0_min

    return float(min(max(K, K_MIN), K_MAX)), float(th0_deg)


def fit_lj(centers: np.ndarray, pmf: np.ndarray,
           g_r: np.ndarray | None = None,
           temperature: float = 300.0) -> tuple[float, float]:
    """
    Fit LJ 12-6 parameters to PMF from Boltzmann inversion of g(r).
    Returns (epsilon [kcal/mol], sigma [Å]).

    Strategy
    --------
    LJ 12-6 has an extremely steep r^{-12} repulsion.  When the CG bead
    g(r) rises gradually at short r (soft repulsion), including the repulsive
    wall in the fit causes curve_fit to choose ε→0, σ→∞ to minimize residuals
    from the mismatch.  The fix is to fit only the attractive well region
    (PMF ≤ well_cutoff) and use the direct Boltzmann estimate
    ε = kBT·ln(g_max), σ = r_min/2^{1/6} as a reliable fallback.

    Ghost counts at very short r (bonded-pair residuals, PBC artifacts) are
    excluded via g_r-based onset detection.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit

    # Thermal energy at the simulation temperature. Used by the direct
    # Boltzmann estimate (eps = kBT * ln(g_max)) and the well-width mask.
    # Previously hard-coded to 300 K, which silently biased epsilon whenever
    # the user requested a different operating temperature.
    _kBT = KB_KCAL * float(temperature)

    valid = np.isfinite(pmf) & (centers > 0)
    if valid.sum() < 5:
        return 0.1, 3.5

    # ── g(r)-based onset: skip isolated ghost bins at very short r ───────────
    if g_r is not None:
        g_sm = gaussian_filter1d(
            np.where(np.isfinite(g_r), g_r, 0.0), sigma=2, mode='nearest')
        r_onset = None
        for i in range(len(centers) - 4):
            if all(g_sm[i:i+4] > 0.05) and centers[i] > 1.0:
                r_onset = centers[i]
                break
        r_onset = r_onset or 2.0
    else:
        r_onset = 2.0

    physical = valid & (centers >= r_onset)
    if physical.sum() < 5:
        physical = valid & (centers > 0.5)

    x = centers[physical]
    y = pmf[physical]

    # ── No-well detection ─────────────────────────────────────────────────────
    pmf_min = float(np.nanmin(y))
    if pmf_min >= -0.01:
        sigma_est = float(np.clip(
            (x[y <= 0.5][0] * 0.89) if (y <= 0.5).any() else 4.0, 2.0, 10.0))
        warnings.warn(
            f"fit_lj: no attractive well (min={pmf_min:.3f} kcal/mol). "
            f"Setting ε=0.1, σ={sigma_est:.2f} Å.")
        return 0.1, sigma_est

    # ── Direct Boltzmann estimate (always computed as fallback) ───────────────
    # ε = kBT·ln(g_max)  from U(r_min) = -kBT·ln(g(r_min)) ≈ -ε for LJ
    # σ = r_min / 2^{1/6}
    if g_r is not None:
        g_phys = np.where(physical, g_r, 0.0)
        g_max  = float(g_phys.max())
        r_gmax = float(centers[np.argmax(g_phys)])
    else:
        r_gmax = float(x[np.argmin(y)])
        g_max  = float(np.exp(-pmf_min / _kBT))
    eps_direct = max(0.01, _kBT * np.log(max(g_max, 1.001)))
    sig_direct = float(np.clip(r_gmax / 2**(1/6), 2.0, 12.0))

    # ── Well-only curve_fit (avoid mismatch with LJ repulsion wall) ──────────
    # Strategy: narrow the fitting region to the PMF minimum neighbourhood
    # (±WELL_HALF_WIDTH Å around r_min, AND PMF ≤ well_cutoff).
    # For polymer CG, g(r) often stays > 1 over a long range due to chain
    # connectivity, so the naive criterion (PMF < pmf_min + 0.5) can capture
    # 90%+ of all bins and cause curve_fit to flatten ε artificially.
    WELL_HALF_WIDTH = 2.0   # Å — typical LJ well half-width for CG beads
    r_pmf_min  = float(centers[physical][np.argmin(pmf[physical])])
    well_mask  = (physical
                  & (pmf  <= pmf_min + 0.5)
                  & (centers >= r_pmf_min - WELL_HALF_WIDTH)
                  & (centers <= r_pmf_min + WELL_HALF_WIDTH))
    if well_mask.sum() < 4:
        well_mask = physical & (pmf <= pmf_min + 0.5)
    if well_mask.sum() < 4:
        well_mask = physical & (pmf <= pmf_min + 2 * _kBT)
    if well_mask.sum() < 4:
        return eps_direct, sig_direct   # not enough well points → direct estimate

    xw = centers[well_mask]
    yw = pmf[well_mask]
    sig0  = float(np.clip(xw[np.argmin(yw)] / 2**(1/6), 2.0, 12.0))
    eps0  = max(0.01, -float(np.nanmin(yw)))

    def lj(rv, eps, sig):
        sr6 = (sig / rv) ** 6
        return 4 * eps * (sr6**2 - sr6)

    try:
        popt, _ = curve_fit(lj, xw, yw, p0=[eps0, sig0],
                            bounds=([0, 0.5], [100, 15]), maxfev=10000)
        eps_fit, sig_fit = float(popt[0]), float(popt[1])
        if not (2.0 <= sig_fit <= 12.0) or not np.isfinite(sig_fit) or eps_fit < 1e-4:
            raise ValueError(f"unphysical: ε={eps_fit:.4f}, σ={sig_fit:.4f}")
        # Guard: if curve_fit gives less than 60% of the direct Boltzmann
        # estimate the well was too broad or noisy → trust the direct estimate.
        if eps_fit < 0.6 * eps_direct:
            warnings.warn(
                f"fit_lj: curve_fit ε={eps_fit:.4f} < 0.6×ε_direct={eps_direct:.4f}. "
                f"Falling back to direct Boltzmann estimate.")
            return eps_direct, sig_direct
        return eps_fit, sig_fit
    except Exception:
        # curve_fit failed or unphysical → use direct Boltzmann estimate
        return eps_direct, sig_direct


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# COHESIVE-ENERGY MATCHING FOR LJ EPSILON
# ─────────────────────────────────────────────────────────────────────────────
# Direct Boltzmann inversion of g(r) yields the POTENTIAL OF MEAN FORCE, not the
# pair potential. In a dense liquid the PMF well is shallow (it already contains
# the cancelling push of all the other neighbours), so eps taken from the PMF
# depth is typically 5-10x too weak. A CG melt built that way has kT >> eps, sits
# above its critical temperature, and simply evaporates -- the CG box expands and
# the density collapses.
#
# Energy matching fixes the depth while keeping the RDF-derived sigma: choose eps
# so the CG pair potential reproduces the all-atom INTERMOLECULAR cohesive energy
# per bead at the target (all-atom) density,
#
#     U/N = (1/2) rho ∫ g(r) u_LJ(r; eps, sigma) 4*pi*r^2 dr  =  U_AA/N
#
# which is linear in eps and therefore needs no iteration.

_COULOMB_KCAL = 332.06371        # kcal·Å/(mol·e²)


def _read_data_sections(datafile_path: str) -> dict:
    """Minimal LAMMPS 'full' data-file reader: box, Pair Coeffs, Atoms."""
    txt = open(datafile_path).read()

    box = []
    for ax in "xyz":
        m = re.search(rf"([-\d.eE+]+)\s+([-\d.eE+]+)\s+{ax}lo\s+{ax}hi", txt)
        if not m:
            raise ValueError(f"No {ax}lo/{ax}hi line in {datafile_path}")
        box.append(float(m.group(2)) - float(m.group(1)))

    def _sect(name: str) -> list[str]:
        i = txt.find("\n" + name)
        if i < 0:
            return []
        i = txt.index("\n", i + 1)
        out, started = [], False
        for line in txt[i:].splitlines():
            body = line.split("#")[0].strip()
            if not body:
                if started:
                    break
                continue
            # a new section header (starts with a letter) ends this one
            if body[0].isalpha():
                break
            started = True
            out.append(body)
        return out

    pair = {}
    for line in _sect("Pair Coeffs"):
        f = line.split()
        if len(f) >= 3:
            pair[int(f[0])] = (float(f[1]), float(f[2]))       # eps, sigma

    atoms = []
    for line in _sect("Atoms"):
        f = line.split()
        if len(f) >= 7:
            atoms.append([float(x) for x in f[:7]])            # id mol type q x y z
    if not atoms:
        raise ValueError(f"No Atoms section parsed from {datafile_path}")

    arr = np.array(sorted(atoms, key=lambda r: r[0]))
    return {"L": np.array(box), "pair": pair, "mol": arr[:, 1].astype(int),
            "type": arr[:, 2].astype(int), "q": arr[:, 3], "xyz": arr[:, 4:7]}


def compute_intermolecular_energy(datafile_path: str, cutoff: float = 12.0,
                                  mixing: str = "geometric") -> dict:
    """
    All-atom INTERMOLECULAR (different-molecule) LJ + Coulomb energy of a frame.

    Only cross-molecule pairs are summed, so the result is the cohesive energy
    that holds the melt together -- exactly what the CG pair potential has to
    reproduce. Intramolecular terms are excluded because in the CG model they
    are carried by the bond and angle potentials.

    Parameters
    ----------
    datafile_path : LAMMPS 'full' data file with Pair Coeffs and charges
                    (e.g. AAEQfin.data)
    cutoff        : pair cutoff in Å (should match the AA run)
    mixing        : "geometric" (OPLS) or "lorentz-berthelot" (sigma arithmetic)

    Returns
    -------
    dict: {"E_lj", "E_coul", "E_total", "n_molecules", "cutoff"}  [kcal/mol]
    """
    d   = _read_data_sections(datafile_path)
    L   = d["L"]; mol = d["mol"]; q = d["q"]
    eps = np.array([d["pair"][t][0] for t in d["type"]])
    sig = np.array([d["pair"][t][1] for t in d["type"]])

    xyz = d["xyz"] - np.floor(d["xyz"] / L) * L
    ncell = np.maximum((L / cutoff).astype(int), 1)
    cell_size = L / ncell
    cidx = (xyz // cell_size).astype(int) % ncell

    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, c in enumerate(map(tuple, cidx)):
        cells[c].append(i)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]

    E_lj = E_coul = 0.0
    for c, ids in cells.items():
        neigh: list[int] = []
        for off in offsets:
            cc = tuple((np.array(c) + off) % ncell)
            if cc in cells:
                neigh.extend(cells[cc])
        neigh_arr = np.array(neigh)
        for i in ids:
            j = neigh_arr[neigh_arr > i]
            j = j[mol[j] != mol[i]]                    # intermolecular only
            if len(j) == 0:
                continue
            dr = xyz[j] - xyz[i]
            dr -= np.round(dr / L) * L
            r = np.sqrt((dr * dr).sum(1))
            m = (r < cutoff) & (r > 0)
            if not m.any():
                continue
            r, jj = r[m], j[m]
            e_ij = np.sqrt(eps[i] * eps[jj])
            s_ij = (np.sqrt(sig[i] * sig[jj]) if mixing == "geometric"
                    else 0.5 * (sig[i] + sig[jj]))
            sr6 = (s_ij / r) ** 6
            E_lj   += float((4.0 * e_ij * (sr6 * sr6 - sr6)).sum())
            E_coul += float((_COULOMB_KCAL * q[i] * q[jj] / r).sum())

    return {"E_lj": E_lj, "E_coul": E_coul, "E_total": E_lj + E_coul,
            "n_molecules": int(mol.max()), "cutoff": cutoff}


def epsilon_from_cohesive_energy(
    rdf_centers: np.ndarray,
    g_r: np.ndarray,
    sigma: float,
    number_density: float,
    u_target_per_bead: float,
    r_cut: float = 10.0,
) -> float:
    """
    Solve  U/N = (1/2) rho ∫ g(r) u_LJ(r) 4*pi*r^2 dr  =  u_target_per_bead
    for eps, with sigma held at its RDF-derived value.

    Parameters
    ----------
    rdf_centers       : bin centres of g(r) [Å]
    g_r               : radial distribution function of the mapped CG beads
    sigma             : LJ sigma from the RDF fit [Å]
    number_density    : TARGET bead number density [beads/Å³] (from the AA melt)
    u_target_per_bead : AA intermolecular energy per bead [kcal/mol] (negative)
    r_cut             : CG pair cutoff [Å]

    Returns
    -------
    eps [kcal/mol]; 0.0 if the integral is degenerate.
    """
    r = np.asarray(rdf_centers, dtype=float)
    g = np.asarray(g_r, dtype=float)
    m = (r > 0) & (r < r_cut) & np.isfinite(g)
    if m.sum() < 5 or number_density <= 0:
        return 0.0
    dr = float(np.mean(np.diff(r[m]))) if m.sum() > 1 else 0.0
    if dr <= 0:
        return 0.0
    u_over_eps = 4.0 * ((sigma / r[m]) ** 12 - (sigma / r[m]) ** 6)
    I = 0.5 * number_density * float(np.sum(g[m] * u_over_eps * 4.0 * np.pi * r[m] ** 2) * dr)
    if I >= 0:          # no net attraction in the integral -> cannot solve
        return 0.0
    return float(u_target_per_bead / I)


def _apply_energy_matching(
    results_lj: dict,
    rdf_data: dict,
    aa_data_path: str,
    n_beads_total: int,
    target_density_gcc: float = None,
    bead_mass_amu: float = None,
    r_cut: float = 10.0,
    epsilon_scale: float = 1.0,
) -> None:
    """
    Rescale every LJ epsilon in `results_lj` (in place) so the CG pair potential
    reproduces the all-atom intermolecular cohesive energy per bead.

    sigma is left untouched -- the RDF locates the packing distance well; it is
    only the WELL DEPTH that Boltzmann inversion gets wrong.

    With several bead types all epsilons are scaled by one common factor, which
    preserves the relative chemistry the RDFs captured while fixing the overall
    cohesion.

    The matching integral is evaluated on the ALL-ATOM structure g_AA(r), but the
    CG melt relaxes to a denser structure once it runs and picks up far more
    cohesion than the integral predicted (observed: 3-4x). Energy matching is
    therefore an UPPER BOUND on epsilon, just as Boltzmann inversion is a lower
    bound; no one-shot integral closes the gap, because it is a feedback between
    the potential and the structure it produces.

    That gap is closed downstream by calibrator.calibrate_epsilon_to_density,
    which searches a scalar multiplier against the AA density using short CG
    runs. The value produced here is its starting point -- and a good one:
    starting from the Boltzmann-inversion epsilon instead would put the search
    deep in the gas phase, where density barely responds to epsilon and the
    secant step is useless.

    `epsilon_scale` is an extra fixed multiplier applied on top, for runs with
    calibration switched off. The notebook passes 1.0 whenever calibration is
    enabled, so the two never fight.
    """
    coh = compute_intermolecular_energy(aa_data_path)
    u_per_bead = coh["E_total"] / max(n_beads_total, 1)

    d = _read_data_sections(aa_data_path)
    V_A3 = float(np.prod(d["L"]))
    if target_density_gcc is None:
        # density of the AA frame we just read (g/cm3), independent of any log
        masses = _masses_by_type(aa_data_path)
        total_mass = float(sum(masses.get(t, 0.0) for t in d["type"]))
        target_density_gcc = total_mass / 6.02214076e23 / (V_A3 * 1e-24)
    if bead_mass_amu is None:
        masses = _masses_by_type(aa_data_path)
        total_mass = float(sum(masses.get(t, 0.0) for t in d["type"]))
        bead_mass_amu = total_mass / max(n_beads_total, 1)

    rho_n = target_density_gcc / bead_mass_amu * 6.02214076e23 / 1e24   # beads/A^3

    print("\n[Potential-Multi] Energy matching (epsilon from AA cohesive energy)")
    print(f"    AA intermolecular E : {coh['E_total']:.1f} kcal/mol "
          f"(LJ {coh['E_lj']:.1f} + Coul {coh['E_coul']:.1f})")
    print(f"    per CG bead         : {u_per_bead:.3f} kcal/mol")
    print(f"    target density      : {target_density_gcc:.4f} g/cm3 "
          f"-> {rho_n:.5f} beads/A^3   (bead mass {bead_mass_amu:.2f} amu)")

    # One common scale factor, derived from the type pair with the most data.
    scales = []
    for key, res in results_lj.items():
        if key not in rdf_data:
            continue
        centers, g_r = rdf_data[key]
        eps_new = epsilon_from_cohesive_energy(
            centers, g_r, res["sigma"], rho_n, u_per_bead, r_cut=r_cut)
        if eps_new > 0:
            scales.append(eps_new / max(res["epsilon"], 1e-9))
    if not scales:
        raise ValueError("no usable RDF for the matching integral")

    scale = float(np.median(scales))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"bad scale factor {scale}")
    if scale > 50:
        print(f"    WARNING: scale factor {scale:.1f} is implausibly large; "
              f"capping at 50. Check the mapping and the AA frame.")
        scale = 50.0

    if epsilon_scale != 1.0:
        print(f"    empirical epsilon_scale : x{epsilon_scale:.3f} "
              f"(energy matching alone overshoots; see docstring)")
    total = scale * float(epsilon_scale)

    for key, res in results_lj.items():
        old = res["epsilon"]
        res["epsilon"] = float(round(old * total, 6))
        print(f"    LJ {key}: eps {old:.6f} -> {res['epsilon']:.6f} kcal/mol "
              f"(x{scale:.2f} matching x{epsilon_scale:.3f} scale), "
              f"sigma {res['sigma']:.4f} A unchanged")

    kT_300 = KB_KCAL * 300.0
    worst = min(r["epsilon"] for r in results_lj.values())
    print(f"    T* = kT(300K)/eps = {kT_300 / max(worst, 1e-9):.2f}  "
          f"(above ~1.3 the melt cannot condense; this is a starting point, "
          f"density calibration refines it)")


def _masses_by_type(datafile_path: str) -> dict:
    """{atom_type: mass} from a LAMMPS data file Masses section."""
    txt = open(datafile_path).read()
    i = txt.find("\nMasses")
    if i < 0:
        return {}
    i = txt.index("\n", i + 1)
    out, started = {}, False
    for line in txt[i:].splitlines():
        body = line.split("#")[0].strip()
        if not body:
            if started:
                break
            continue
        if body[0].isalpha():
            break
        f = body.split()
        if len(f) >= 2:
            started = True
            out[int(f[0])] = float(f[1])
    return out


def compute_cg_potentials(
    trj_path:     str,
    aa_data_path: str,
    poly_def:     dict,
    bead_map_def: dict,
    n_chains:     int,
    n_monomers:   int,
    temperature:  float = 300.0,
    frame_stride: int   = 1,
    bond_bins:    int   = 200,
    angle_bins:   int   = 180,
    rdf_bins:     int   = 500,
    rdf_frac:     float = 0.35,
    output_dir:   str   = None,
) -> dict:
    """
    Full pipeline: trajectory → distributions → Boltzmann inversion → fit.

    Returns
    -------
    {
      "bond":  {"r0": Å, "k": kcal/mol/Å²},
      "angle": {"theta0": deg, "k": kcal/mol/rad²},
      "lj":    {"epsilon": kcal/mol, "sigma": Å},
      "csv":   {"bond": path, "angle": path, "rdf": path}  (if output_dir given)
    }
    """
    kBT = KB_KCAL * temperature

    print("[Potential] Building bead list from AA data...")
    bead_list, atom_masses, mol_bead_ids = build_bead_list(
        aa_data_path, poly_def, bead_map_def, n_chains, n_monomers
    )
    bond_pairs, angle_trips = _bond_angle_pairs(mol_bead_ids)
    n_beads = len(bead_list)
    print(f"[Potential]   {n_beads} beads, {len(bond_pairs)} bond pairs, "
          f"{len(angle_trips)} angle triplets")

    # Histogram accumulators
    bond_edges  = np.linspace(0.0, 15.0,  bond_bins  + 1)
    angle_edges = np.linspace(0.0, 180.0, angle_bins + 1)
    bond_hist   = np.zeros(bond_bins,  dtype=np.int64)
    angle_hist  = np.zeros(angle_bins, dtype=np.int64)
    rdf_hist    = np.zeros(rdf_bins,   dtype=np.int64)

    rdf_rmax  = None
    n_frames  = 0
    V_sum     = 0.0

    print("[Potential] Reading trajectory frames...")
    for fi, (ts, box, coords) in enumerate(iter_frames(trj_path)):
        if fi % frame_stride != 0:
            continue

        L    = box[:, 1] - box[:, 0]
        pos  = bead_positions_from_frame(bead_list, atom_masses, coords, box)

        if len(bond_pairs) > 0:
            bl = _compute_bonds(pos, bond_pairs, L)
            bond_hist += np.histogram(bl, bins=bond_edges)[0]

        if len(angle_trips) > 0:
            ag = _compute_angles(pos, angle_trips, L)
            angle_hist += np.histogram(ag, bins=angle_edges)[0]

        r_max = rdf_frac * float(np.min(L))
        rdf_rmax = r_max if rdf_rmax is None else min(rdf_rmax, r_max)
        h, _ = _compute_rdf(pos, L, rdf_rmax, rdf_bins)
        rdf_hist += h

        V_sum += float(np.prod(L))
        n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames found in trajectory.")
    print(f"[Potential] Used {n_frames} frames.")

    # ── Normalise ────────────────────────────────────────────────────────────
    bond_centers  = 0.5 * (bond_edges[:-1]  + bond_edges[1:])
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    dr_bond  = bond_edges[1]  - bond_edges[0]
    dr_angle = angle_edges[1] - angle_edges[0]

    bond_pdf  = bond_hist  / (bond_hist.sum()  * dr_bond  + 1e-30)
    angle_pdf = angle_hist / (angle_hist.sum() * dr_angle + 1e-30)

    rdf_edges   = np.linspace(0.0, rdf_rmax, rdf_bins + 1)
    rdf_centers = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
    shell_vol   = (4*np.pi/3) * (rdf_edges[1:]**3 - rdf_edges[:-1]**3)
    V_avg       = V_sum / n_frames
    rho         = n_beads / V_avg
    ideal       = n_beads * rho * shell_vol / 2.0
    g_r         = rdf_hist / (ideal * n_frames + 1e-30)

    # ── Boltzmann inversion ──────────────────────────────────────────────────
    pmf_bond  = boltzmann_inversion_bond(bond_centers,  bond_pdf,  kBT)
    pmf_angle = boltzmann_inversion_angle(angle_centers, angle_pdf, kBT)
    pmf_lj    = boltzmann_inversion_rdf(rdf_centers,    g_r,       kBT)

    # ── Fitting ──────────────────────────────────────────────────────────────
    k_bond,  r0      = fit_harmonic_bond(bond_centers,  pmf_bond)
    k_angle, theta0  = fit_harmonic_angle(angle_centers, pmf_angle)
    epsilon, sigma   = fit_lj(rdf_centers, pmf_lj, g_r=g_r, temperature=temperature)

    results = {
        # float() is required: round() keeps numpy.float64, and LangGraph's
        # msgpack checkpointer cannot serialize numpy scalars.
        "bond":  {"r0": float(round(r0, 4)),        "k": float(round(k_bond, 4))},
        "angle": {"theta0": float(round(theta0, 3)), "k": float(round(k_angle, 4))},
        "lj":    {"epsilon": float(round(epsilon, 6)), "sigma": float(round(sigma, 4))},
        "temperature": temperature,
        "n_frames": n_frames,
    }

    # ── Save CSV (optional) ──────────────────────────────────────────────────
    if output_dir:
        import csv, os
        os.makedirs(output_dir, exist_ok=True)
        csv_paths = {}

        def _write_csv(path, headers, rows):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
            return path

        csv_paths["bond"] = _write_csv(
            os.path.join(output_dir, "cg_bond_pdf.csv"),
            ["r_A", "pdf", "pmf_kcal_mol"],
            zip(bond_centers, bond_pdf,
                np.where(np.isfinite(pmf_bond), pmf_bond, 0.0))
        )
        csv_paths["angle"] = _write_csv(
            os.path.join(output_dir, "cg_angle_pdf.csv"),
            ["theta_deg", "pdf", "pmf_kcal_mol"],
            zip(angle_centers, angle_pdf,
                np.where(np.isfinite(pmf_angle), pmf_angle, 0.0))
        )
        csv_paths["rdf"] = _write_csv(
            os.path.join(output_dir, "cg_rdf.csv"),
            ["r_A", "g_r", "pmf_kcal_mol"],
            zip(rdf_centers, g_r,
                np.where(np.isfinite(pmf_lj), pmf_lj, 0.0))
        )
        results["csv"] = csv_paths
        print(f"[Potential] CSVs saved to {output_dir}")

    print(f"\n[Potential] Results:")
    print(f"  Bond  : r0={r0:.4f} Å,   k={k_bond:.4f} kcal/mol/Å²")
    print(f"  Angle : θ0={theta0:.3f}°,  k={k_angle:.4f} kcal/mol/rad²")
    print(f"  LJ    : ε={epsilon:.6f} kcal/mol,  σ={sigma:.4f} Å")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TYPE POTENTIAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def build_bead_list_typed(
    aa_data_path: str,
    poly_def:     dict,
    bead_map_def: dict,
    n_chains:     int,
    n_monomers:   int,
) -> tuple:
    """
    Like build_bead_list but returns bead type info and proper connectivity.

    Returns:
      bead_list_simple : [(mol_id, [atom_ids]), ...]          for bead_positions_from_frame
      bead_type_arr    : np.ndarray of bead_type_id per bead
      atom_masses      : {atom_id: mass}
      mol_bead_structure : per mol: {"backbone": [idx,...], "pendants": {bb_idx: [p_idx,...]}}
    """
    from mapper import read_aa_data, build_mol_adj, linear_chain_order

    atoms, bonds_aa, box, masses_map = read_aa_data(aa_data_path)
    mol_atoms, adj = build_mol_adj(atoms, bonds_aa)

    from polymer import normalize_poly_def, compute_monomer_offsets
    poly_def_n    = normalize_poly_def(poly_def, n_monomers)
    sequence      = poly_def_n["sequence"]
    monomer_types_map = poly_def_n["monomer_types"]
    bead_types    = bead_map_def["bead_types"]
    has_mono_type = any("monomer_type" in bt for bt in bead_types)
    max_mpb       = max((bt.get("monomers_per_bead") or bt.get("repeating_monomers",1) or 1) for bt in bead_types)

    backbone_bt_id = next(
        (bt["bead_type_id"] for bt in bead_types if bt.get("is_backbone", False)),
        bead_types[0]["bead_type_id"]
    )

    bead_list_simple = []
    bead_type_list   = []
    atom_masses      = {aid: masses_map.get(a.atype, 12.011) for aid, a in atoms.items()}
    mol_bead_structure = []

    for mid in sorted(mol_atoms.keys()):
        atom_order = linear_chain_order(mol_atoms[mid], adj)
        n_chain    = len(atom_order)

        offsets_chain = compute_monomer_offsets(sequence, monomer_types_map)
        monomer_groups = []
        for mi, lbl in enumerate(sequence):
            s, e = offsets_chain[mi], offsets_chain[mi+1]
            if e <= n_chain:
                monomer_groups.append((lbl, atom_order[s:e]))

        backbone_indices = []
        pendants_map     = {}

        # ── Detect topology mode ──────────────────────────────────────────────
        # "linear copolymer": every monomer contributes exactly ONE bead to the
        #   main chain (e.g. PE+PP). All beads are backbone, bonds follow the
        #   actual monomer sequence.
        # "backbone+pendant": a single monomer type has MULTIPLE bead types
        #   (e.g. PS: backbone CH2 + ring bead). One bead is backbone, others
        #   are pendants attached to it.
        if has_mono_type:
            beads_per_mono_type = {
                lbl: len([bt for bt in bead_types if bt.get("monomer_type") == lbl])
                for lbl in set(lbl for lbl, _ in monomer_groups)
            }
            is_linear_copoly = all(n == 1 for n in beads_per_mono_type.values())
        else:
            is_linear_copoly = False

        mono_idx = 0
        while mono_idx < len(monomer_groups):
            group_backbone_idx = None
            cur_lbl    = monomer_groups[mono_idx][0]
            applicable = ([bt for bt in bead_types if bt.get("monomer_type") == cur_lbl]
                         or [bt for bt in bead_types if "monomer_type" not in bt]) if has_mono_type else bead_types

            for bt in applicable:
                lids = bt.get("atom_local_ids") or bt.get("local_ids_in_monomer", [])
                mpb  = bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1
                bead_atom_ids = []
                for m_off in range(mpb):
                    mi2 = mono_idx + m_off
                    if mi2 >= len(monomer_groups): break
                    _, mg = monomer_groups[mi2]
                    for lid in lids:
                        idx2 = lid - 1
                        if 0 <= idx2 < len(mg):
                            bead_atom_ids.append(mg[idx2])

                if not bead_atom_ids:
                    continue

                global_idx = len(bead_list_simple)
                bead_list_simple.append((mid, bead_atom_ids))
                bead_type_list.append(bt["bead_type_id"])

                # Linear copolymer: every bead is on the main chain
                if is_linear_copoly:
                    backbone_indices.append(global_idx)

                # Backbone+pendant: classify by bead_type_id or is_backbone flag
                elif bt.get("is_backbone", False) or bt["bead_type_id"] == backbone_bt_id:
                    backbone_indices.append(global_idx)
                    group_backbone_idx = global_idx
                else:
                    if group_backbone_idx is not None:
                        if group_backbone_idx not in pendants_map:
                            pendants_map[group_backbone_idx] = []
                        pendants_map[group_backbone_idx].append(global_idx)

            mono_idx += max_mpb

        mol_bead_structure.append({
            "backbone": backbone_indices,
            "pendants": pendants_map,
        })

    return bead_list_simple, np.array(bead_type_list, dtype=int), atom_masses, mol_bead_structure, bonds_aa


def _typed_bond_angle_pairs(bead_type_arr, mol_bead_structure, bead_list_simple=None, bonds_aa=None):
    """
    Build typed bond/angle lists.

    Preferred path (when bead_list_simple + bonds_aa are given): derive bead
    connectivity from the underlying AA bonds — two beads are bonded iff any of
    their atoms are chemically bonded. This is correct for every mapping style
    (linear backbone, monomer-split A-B-A-B, or pendant side beads) and matches
    exactly what the bead mapper writes into the CG data file, so the measured
    angle histograms line up with the topology keys.

    Returns:
      typed_bonds  : [(idx_i, idx_j, (ti, tj)), ...]   ti <= tj
      typed_angles : [(idx_i, idx_j, idx_k, (ti, tj, tk)), ...]  ends sorted
    """
    if bead_list_simple is not None and bonds_aa is not None:
        # Map each AA atom id -> bead index
        atom_to_bead = {}
        for bidx, (_mid, atom_ids) in enumerate(bead_list_simple):
            for a in atom_ids:
                atom_to_bead[a] = bidx

        bead_adj = {}
        for (a, b) in bonds_aa:
            ba, bb = atom_to_bead.get(a), atom_to_bead.get(b)
            if ba is not None and bb is not None and ba != bb:
                bead_adj.setdefault(ba, set()).add(bb)
                bead_adj.setdefault(bb, set()).add(ba)

        typed_bonds = []
        seen = set()
        for bi in sorted(bead_adj):
            for bj in bead_adj[bi]:
                key = (min(bi, bj), max(bi, bj))
                if key in seen:
                    continue
                seen.add(key)
                ti, tj = int(bead_type_arr[bi]), int(bead_type_arr[bj])
                typed_bonds.append((key[0], key[1], (min(ti, tj), max(ti, tj))))

        typed_angles = []
        for bj in sorted(bead_adj):
            nbrs = sorted(bead_adj[bj])
            for x in range(len(nbrs)):
                for y in range(x + 1, len(nbrs)):
                    bi, bk = nbrs[x], nbrs[y]
                    ti = int(bead_type_arr[bi]); tj = int(bead_type_arr[bj]); tk = int(bead_type_arr[bk])
                    ends = sorted((ti, tk))
                    typed_angles.append((bi, bj, bk, (ends[0], tj, ends[1])))
        return typed_bonds, typed_angles

    # ── Fallback: legacy backbone/pendant reconstruction ────────────────────
    typed_bonds  = []
    typed_angles = []

    for mol_str in mol_bead_structure:
        bb   = mol_str["backbone"]
        pend = mol_str["pendants"]

        # backbone-backbone bonds
        for i in range(len(bb) - 1):
            bi, bj = bb[i], bb[i + 1]
            ti, tj = int(bead_type_arr[bi]), int(bead_type_arr[bj])
            pair = (min(ti, tj), max(ti, tj))
            typed_bonds.append((bi, bj, pair))

        # backbone-pendant bonds
        for bi, pidxs in pend.items():
            ti = int(bead_type_arr[bi])
            for pj in pidxs:
                tj = int(bead_type_arr[pj])
                pair = (min(ti, tj), max(ti, tj))
                typed_bonds.append((bi, pj, pair))

        # backbone-backbone-backbone angles
        for i in range(len(bb) - 2):
            bi, bj, bk = bb[i], bb[i + 1], bb[i + 2]
            ti, tj, tk = int(bead_type_arr[bi]), int(bead_type_arr[bj]), int(bead_type_arr[bk])
            typed_angles.append((bi, bj, bk, (ti, tj, tk)))

        # backbone[i]-backbone[i+1]-pendant[backbone[i+1]] angles
        for i in range(len(bb) - 1):
            bj = bb[i + 1]
            bi = bb[i]
            if bj in pend:
                for pk in pend[bj]:
                    ti, tj, tk = int(bead_type_arr[bi]), int(bead_type_arr[bj]), int(bead_type_arr[pk])
                    typed_angles.append((bi, bj, pk, (ti, tj, tk)))

        # pendant[backbone[i]]-backbone[i]-backbone[i+1] angles
        for i in range(len(bb) - 1):
            bi_prev = bb[i]
            bj_next = bb[i + 1]
            if bi_prev in pend:
                for pi in pend[bi_prev]:
                    ti, tj, tk = int(bead_type_arr[pi]), int(bead_type_arr[bi_prev]), int(bead_type_arr[bj_next])
                    typed_angles.append((pi, bi_prev, bj_next, (ti, tj, tk)))

    return typed_bonds, typed_angles


def compute_cg_potentials_multi(
    trj_path:      str,
    aa_data_path:  str,
    poly_def:      dict,
    bead_map_def:  dict,
    cg_topology:   dict,
    n_chains:      int,
    n_monomers:    int,
    temperature:   float = 300.0,
    frame_stride:  int   = 1,
    bond_bins:     int   = 200,
    angle_bins:    int   = 180,
    rdf_bins:      int   = 500,
    rdf_frac:      float = 0.35,
    output_dir:    str   = None,
    rdf_exclude_depth: int = 3,
    energy_matching:   bool = True,
    epsilon_scale:     float = 1.0,
    target_density_gcc: float = None,
    bead_mass_amu:      float = None,
    cg_pair_cutoff:     float = 10.0,
) -> dict:
    """
    Multi-bead-type CG potential computation.

    Extra parameters
    ----------------
    rdf_exclude_depth  : how many bonds out to exclude from the RDF.
                         3 = exclude 1-2/1-3/1-4, matching LAMMPS'
                         default `special_bonds 0.0 0.0 0.0`. Use 1 for the
                         old behaviour (bonded pairs only).
    epsilon_scale      : fixed extra multiplier on the matched epsilon, for use
                         when density calibration is DISABLED. Energy matching
                         is an upper bound (see _apply_energy_matching), so a
                         value below 1 is expected. Leave at 1.0 when
                         calibrator.calibrate_epsilon_to_density will run -- it
                         determines the multiplier by measurement instead.
    energy_matching    : if True, rescale each LJ epsilon so the CG pair
                         potential reproduces the all-atom intermolecular
                         cohesive energy per bead. Boltzmann inversion alone
                         gives the PMF depth, which is far too shallow and
                         makes the CG melt evaporate.
    target_density_gcc : AA equilibrium density [g/cm³] used for the matching
                         integral. Defaults to the density of aa_data_path.
    bead_mass_amu      : mass of one CG bead [amu]; defaults to the mapped mass.
    cg_pair_cutoff     : pair cutoff used in the CG run [Å].

    cg_topology = {
        "bond_type_pairs":    [[1,1], [1,2], ...],
        "angle_type_triplets":[[1,1,1], [1,1,2], ...],
        "lj_type_pairs":      [[1,1], [1,2], [2,2], ...]
    }

    Returns {
        "bond":   {(ti,tj):  {"r0":..., "k":...}},
        "angle":  {(ti,tj,tk):{"theta0":..., "k":...}},
        "lj":     {(ti,tj):  {"epsilon":..., "sigma":...}},
        "bond_type_order":   [(ti,tj), ...],
        "angle_type_order":  [(ti,tj,tk), ...],
        "lj_type_order":     [(ti,tj), ...],
    }
    """
    kBT = KB_KCAL * temperature

    print("[Potential-Multi] Building typed bead list...")
    bead_list_simple, bead_type_arr, atom_masses, mol_bead_structure, bonds_aa = \
        build_bead_list_typed(aa_data_path, poly_def, bead_map_def, n_chains, n_monomers)

    n_beads = len(bead_list_simple)
    print(f"[Potential-Multi]   {n_beads} beads across {n_chains} chains")

    typed_bonds, typed_angles = _typed_bond_angle_pairs(
        bead_type_arr, mol_bead_structure, bead_list_simple, bonds_aa)
    print(f"[Potential-Multi]   {len(typed_bonds)} bond pairs, {len(typed_angles)} angle triplets")

    # Normalise topology keys. Bonds sorted; angles use (sorted-ends, centre)
    # so the keys match the normalised triplets produced for typed_angles
    # (otherwise a (2,1,1) topology key never matches a (1,1,2) measurement).
    def _norm_ang_key(t):
        e = tuple(sorted((t[0], t[2])))
        return (e[0], t[1], e[1])
    bond_type_pairs    = [tuple(sorted(p)) for p in cg_topology.get("bond_type_pairs", [])]
    angle_type_triplets= [_norm_ang_key(tuple(t)) for t in cg_topology.get("angle_type_triplets", [])]
    # de-duplicate after normalisation, preserving order
    bond_type_pairs    = list(dict.fromkeys(bond_type_pairs))
    angle_type_triplets= list(dict.fromkeys(angle_type_triplets))
    lj_type_pairs      = [tuple(sorted(p)) for p in cg_topology.get("lj_type_pairs", [])]
    lj_type_pairs      = list(dict.fromkeys(lj_type_pairs))

    bond_edges  = np.linspace(0.0, 15.0,  bond_bins  + 1)
    angle_edges = np.linspace(0.0, 180.0, angle_bins + 1)

    bond_hists  = {k: np.zeros(bond_bins,  dtype=np.int64) for k in bond_type_pairs}
    angle_hists = {k: np.zeros(angle_bins, dtype=np.int64) for k in angle_type_triplets}
    lj_hists    = {k: np.zeros(rdf_bins,   dtype=np.int64) for k in lj_type_pairs}

    # ── Pre-build bonded bead index set for RDF exclusion ────────────────────
    # Bonded pairs (backbone-backbone and backbone-pendant) share a covalent-like
    # bond in the CG model.  Including them in the RDF contaminates the non-bonded
    # g(r) with a large spurious peak at the bond distance, making LJ ε/σ wrong.
    bonded_set: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = defaultdict(set)
    for mol_str in mol_bead_structure:
        bb   = mol_str["backbone"]
        pend = mol_str["pendants"]
        for i in range(len(bb) - 1):
            a, b = bb[i], bb[i + 1]
            bonded_set.add((min(a, b), max(a, b)))
            adjacency[a].add(b); adjacency[b].add(a)
        for bi, pidxs in pend.items():
            for pj in pidxs:
                bonded_set.add((min(bi, pj), max(bi, pj)))
                adjacency[bi].add(pj); adjacency[pj].add(bi)

    # LAMMPS applies `special_bonds 0.0 0.0 0.0` by default, i.e. 1-2, 1-3 AND
    # 1-4 neighbours feel NO pair interaction. The RDF must therefore be built
    # over exactly the same pairs, otherwise g(r) is fitted on pairs the CG run
    # will never evaluate. This matters a lot: with a typical backbone the 1-3
    # distance (2*r0*sin(theta0/2)) lands right on top of the first coordination
    # shell, so leaving those pairs in distorts sigma and epsilon precisely where
    # they are determined.
    excluded_set: set[tuple[int, int]] = set(bonded_set)
    if rdf_exclude_depth >= 2:
        for src, nbrs in list(adjacency.items()):
            # BFS out to `rdf_exclude_depth` bonds from every bead
            frontier, seen = {src}, {src}
            for _ in range(rdf_exclude_depth):
                nxt = set()
                for u in frontier:
                    nxt |= adjacency[u]
                nxt -= seen
                for v in nxt:
                    excluded_set.add((min(src, v), max(src, v)))
                seen |= nxt
                frontier = nxt
    n_extra = len(excluded_set) - len(bonded_set)
    print(f"[Potential-Multi]   {len(bonded_set)} bonded pairs excluded from RDF"
          + (f" (+{n_extra} 1-3/1-4 pairs, special_bonds-consistent)"
             if n_extra else ""))
    bonded_set = excluded_set

    n_frames = 0
    rdf_rmax = None
    V_sum    = 0.0

    print("[Potential-Multi] Reading trajectory frames...")
    for fi, (ts, box, coords) in enumerate(iter_frames(trj_path)):
        if fi % frame_stride != 0:
            continue

        L   = box[:, 1] - box[:, 0]
        pos = bead_positions_from_frame(bead_list_simple, atom_masses, coords, box)

        # ── bonds ─────────────────────────────────────────────────────────────
        for (bi, bj, btype) in typed_bonds:
            if btype not in bond_hists:
                continue
            dr = _mic(pos[bj] - pos[bi], L)
            r  = float(np.linalg.norm(dr))
            if 0.0 < r < 15.0:
                bond_hists[btype][min(int(r / bond_edges[1]), bond_bins - 1)] += 1

        # ── angles ────────────────────────────────────────────────────────────
        for (bi, bj, bk, atype) in typed_angles:
            if atype not in angle_hists:
                continue
            v1 = _mic(pos[bi] - pos[bj], L)
            v2 = _mic(pos[bk] - pos[bj], L)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cos_val = np.dot(v1, v2) / (n1 * n2)
            angle   = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
            angle_hists[atype][min(int(angle / angle_edges[1]), angle_bins - 1)] += 1

        # ── LJ / RDF per type pair  (bonded pairs excluded) ──────────────────
        r_max    = rdf_frac * float(np.min(L))
        rdf_rmax = r_max if rdf_rmax is None else min(rdf_rmax, r_max)
        rdf_edges_frame = np.linspace(0.0, rdf_rmax, rdf_bins + 1)

        for (ti, tj) in lj_type_pairs:
            idx_i = np.where(bead_type_arr == ti)[0]
            idx_j = np.where(bead_type_arr == tj)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue

            pos_i = pos[idx_i]
            pos_j = pos[idx_j]

            if ti == tj:
                n = len(pos_i)
                for k in range(n - 1):
                    gi = int(idx_i[k])
                    for l_off in range(k + 1, n):
                        gj = int(idx_i[l_off])
                        if (min(gi, gj), max(gi, gj)) in bonded_set:
                            continue
                        dr_single = _mic(pos_j[l_off] - pos_i[k], L)
                        r_single  = float(np.linalg.norm(dr_single))
                        if 0.0 < r_single < rdf_rmax:
                            bin_idx = min(int(r_single / (rdf_rmax / rdf_bins)), rdf_bins - 1)
                            lj_hists[(ti, tj)][bin_idx] += 1
            else:
                for k in range(len(pos_i)):
                    gi = int(idx_i[k])
                    for l_off in range(len(idx_j)):
                        gj = int(idx_j[l_off])
                        if (min(gi, gj), max(gi, gj)) in bonded_set:
                            continue
                        dr_single = _mic(pos_j[l_off] - pos_i[k], L)
                        r_single  = float(np.linalg.norm(dr_single))
                        if 0.0 < r_single < rdf_rmax:
                            bin_idx = min(int(r_single / (rdf_rmax / rdf_bins)), rdf_bins - 1)
                            lj_hists[(ti, tj)][bin_idx] += 1

        V_sum += float(np.prod(L))
        n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames found in trajectory.")
    print(f"[Potential-Multi] Used {n_frames} frames.")

    # ── Normalise & fit ───────────────────────────────────────────────────────
    bond_centers  = 0.5 * (bond_edges[:-1]  + bond_edges[1:])
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    dr_bond       = bond_edges[1]  - bond_edges[0]
    dr_angle      = angle_edges[1] - angle_edges[0]

    rdf_edges_f  = np.linspace(0.0, rdf_rmax, rdf_bins + 1)
    rdf_centers  = 0.5 * (rdf_edges_f[:-1] + rdf_edges_f[1:])
    shell_vol    = (4 * np.pi / 3) * (rdf_edges_f[1:]**3 - rdf_edges_f[:-1]**3)
    V_avg        = V_sum / n_frames

    results_bond  = {}
    results_angle = {}
    results_lj    = {}

    for btype, hist in bond_hists.items():
        if hist.sum() == 0:
            print(f"  Bond {btype}: WARNING no data found, skipping fit")
            results_bond[btype] = {"r0": 0.0, "k": 0.0}
            continue
        pdf = hist / (hist.sum() * dr_bond + 1e-30)
        pmf = boltzmann_inversion_bond(bond_centers, pdf, kBT)
        k, r0 = fit_harmonic_bond(bond_centers, pmf)
        results_bond[btype] = {"r0": float(round(r0, 4)), "k": float(round(k, 4))}
        print(f"  Bond {btype}: r0={r0:.4f} Å  k={k:.4f} kcal/mol/Å²")

    for atype, hist in angle_hists.items():
        if hist.sum() == 0:
            print(f"  Angle {atype}: WARNING no data found, skipping fit")
            results_angle[atype] = {"theta0": 0.0, "k": 0.0}
            continue
        pdf = hist / (hist.sum() * dr_angle + 1e-30)
        pmf = boltzmann_inversion_angle(angle_centers, pdf, kBT)
        k, th0 = fit_harmonic_angle(angle_centers, pmf)
        results_angle[atype] = {"theta0": float(round(th0, 3)), "k": float(round(k, 4))}
        print(f"  Angle {atype}: θ0={th0:.3f}°  k={k:.4f} kcal/mol/rad²")

    rdf_for_matching: dict = {}
    for (ti, tj), hist in lj_hists.items():
        if hist.sum() == 0:
            print(f"  LJ {(ti,tj)}: WARNING no data found, skipping fit")
            results_lj[(ti, tj)] = {"epsilon": 0.0, "sigma": 3.5}
            continue
        n_i   = int(np.sum(bead_type_arr == ti))
        n_j   = int(np.sum(bead_type_arr == tj))
        V_avg = V_sum / n_frames
        rdf_edges_f = np.linspace(0.0, rdf_rmax, rdf_bins + 1)
        rdf_centers_f = 0.5 * (rdf_edges_f[:-1] + rdf_edges_f[1:])
        shell_vol_f = (4*np.pi/3) * (rdf_edges_f[1:]**3 - rdf_edges_f[:-1]**3)
        # Correct normalization per type pair
        if ti == tj:
            n_pairs = n_i * (n_i - 1) / 2.0
        else:
            n_pairs = float(n_i * n_j)
        rho_j  = n_j / V_avg
        norm   = n_pairs * rho_j * shell_vol_f / float(n_j) if n_j > 0 else shell_vol_f
        # Simpler: ideal gas count per frame
        ideal  = n_pairs * shell_vol_f / V_avg
        g_r    = hist / (ideal * n_frames + 1e-30)
        pmf    = boltzmann_inversion_rdf(rdf_centers_f, g_r, kBT)
        epsilon, sigma = fit_lj(rdf_centers_f, pmf, g_r=g_r, temperature=temperature)
        results_lj[(ti, tj)] = {"epsilon": float(round(epsilon, 6)), "sigma": float(round(sigma, 4))}
        print(f"  LJ {(ti,tj)}: n_i={n_i} n_j={n_j} ε={epsilon:.6f}  σ={sigma:.4f} Å  (Boltzmann inversion)")
        rdf_for_matching[(ti, tj)] = (rdf_centers_f.copy(), g_r.copy())

    # ── Energy matching: rescale epsilon to the AA cohesive energy ────────────
    if energy_matching and results_lj:
        try:
            _apply_energy_matching(
                results_lj         = results_lj,
                rdf_data           = rdf_for_matching,
                aa_data_path       = aa_data_path,
                n_beads_total      = n_beads,
                target_density_gcc = target_density_gcc,
                bead_mass_amu      = bead_mass_amu,
                r_cut              = cg_pair_cutoff,
                epsilon_scale      = epsilon_scale,
            )
        except Exception as e:
            print(f"[Potential-Multi] Energy matching skipped: {e}")
            print("[Potential-Multi] WARNING: epsilon then comes from the PMF "
                  "depth alone, which underestimates cohesion badly -- expect "
                  "the CG density to collapse.")

    # ── Optional CSV output ───────────────────────────────────────────────────
    if output_dir:
        import csv, os
        os.makedirs(output_dir, exist_ok=True)
        for btype, hist in bond_hists.items():
            pdf = hist / (hist.sum() * dr_bond + 1e-30)
            pmf = boltzmann_inversion_bond(bond_centers, pdf, kBT)
            path = os.path.join(output_dir, f"cg_bond_{btype[0]}_{btype[1]}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["r_A", "pdf", "pmf_kcal_mol"])
                w.writerows(zip(bond_centers, pdf, np.where(np.isfinite(pmf), pmf, 0.0)))
        for atype, hist in angle_hists.items():
            pdf = hist / (hist.sum() * dr_angle + 1e-30)
            pmf = boltzmann_inversion_angle(angle_centers, pdf, kBT)
            path = os.path.join(output_dir, f"cg_angle_{'_'.join(map(str,atype))}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["theta_deg", "pdf", "pmf_kcal_mol"])
                w.writerows(zip(angle_centers, pdf, np.where(np.isfinite(pmf), pmf, 0.0)))

        # ── LJ / RDF CSV ──────────────────────────────────────────────────────
        rdf_edges_csv = np.linspace(0.0, rdf_rmax, rdf_bins + 1)
        rdf_centers_csv = 0.5 * (rdf_edges_csv[:-1] + rdf_edges_csv[1:])
        shell_vol_csv = (4*np.pi/3) * (rdf_edges_csv[1:]**3 - rdf_edges_csv[:-1]**3)
        V_avg_csv = V_sum / n_frames
        for (ti, tj), hist in lj_hists.items():
            n_i = int(np.sum(bead_type_arr == ti))
            n_j = int(np.sum(bead_type_arr == tj))
            if ti == tj:
                n_pairs = n_i * (n_i - 1) / 2.0
            else:
                n_pairs = float(n_i * n_j)
            ideal = n_pairs * shell_vol_csv / V_avg_csv
            g_r   = hist / (ideal * n_frames + 1e-30)
            pmf   = boltzmann_inversion_rdf(rdf_centers_csv, g_r, kBT)
            path  = os.path.join(output_dir, f"cg_rdf_{ti}_{tj}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["r_A", "g_r", "pmf_kcal_mol"])
                w.writerows(zip(rdf_centers_csv, g_r, np.where(np.isfinite(pmf), pmf, 0.0)))

    return {
        "bond":               results_bond,
        "angle":              results_angle,
        "lj":                 results_lj,
        "bond_type_order":    bond_type_pairs,
        "angle_type_order":   angle_type_triplets,
        "lj_type_order":      lj_type_pairs,
        "n_frames":           n_frames,
        "temperature":        temperature,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(output_dir: str, show: bool = False, **_ignored) -> list[str]:
    """
    potential 단계가 세션 폴더에 써 둔 분포 CSV를 그림으로 저장합니다.

        cg_bond_<i>_<j>.csv        r_A, pdf, pmf_kcal_mol
        cg_angle_<i>_<j>_<k>.csv   theta_deg, pdf, pmf_kcal_mol
        cg_rdf_<i>_<j>.csv         r_A, g_r, pmf_kcal_mol

    각 CSV마다 좌: 분포(PDF 또는 g(r)), 우: Boltzmann 역변환 PMF 두 패널을
    그립니다. 파일이 없으면 조용히 빈 리스트를 돌려줍니다.

    Returns
    -------
    저장된 png 경로 리스트
    """
    import glob
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg") if not show else None
    import matplotlib.pyplot as plt

    specs = [
        ("cg_bond_*.csv",  "Bond length (Å)",  "PDF",   "Bond"),
        ("cg_angle_*.csv", "Angle (deg)",      "PDF",   "Angle"),
        ("cg_rdf_*.csv",   "r (Å)",            "g(r)",  "RDF"),
    ]

    saved: list[str] = []
    for pattern, xlabel, ylabel, kind in specs:
        for path in sorted(glob.glob(os.path.join(output_dir, pattern))):
            try:
                with open(path, newline="") as f:
                    rows = list(_csv.reader(f))
                if len(rows) < 2:
                    continue
                cols = list(zip(*[[float(v) for v in r] for r in rows[1:] if len(r) >= 3]))
                if len(cols) < 3:
                    continue
                x, dist, pmf = (np.array(c) for c in cols[:3])

                # PMF는 접근 불가 영역에서 발산하므로 유한한 값만 그린다.
                finite = np.isfinite(pmf) & (dist > 0)

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].plot(x, dist, lw=1.5)
                axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
                axes[0].set_title(f"{kind} distribution")
                axes[0].grid(alpha=0.3)

                if finite.any():
                    axes[1].plot(x[finite], pmf[finite], lw=1.5, color="tab:red")
                axes[1].set_xlabel(xlabel)
                axes[1].set_ylabel("PMF (kcal/mol)")
                axes[1].set_title(f"{kind} PMF (Boltzmann inversion)")
                axes[1].grid(alpha=0.3)

                tag  = os.path.splitext(os.path.basename(path))[0]
                out  = os.path.join(output_dir, tag + "_dist.png")
                fig.suptitle(tag)
                fig.tight_layout()
                fig.savefig(out, dpi=150)
                if show:
                    plt.show()
                plt.close(fig)
                saved.append(out)
                print(f"[Potential] Saved: {out}")
            except Exception as e:
                print(f"[Potential] Could not plot {os.path.basename(path)}: {e}")

    if not saved:
        print(f"[Potential] No distribution CSVs found in {output_dir}")
    return saved


# 이전 이름으로 호출하는 코드를 위한 별칭
plot_potentials = plot_distributions
