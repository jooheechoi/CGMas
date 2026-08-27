"""
polymer.py
==========
Core utilities for CGMas polymer topology definitions.

Supports homopolymers and copolymers (block, alternating, random,
periodic, graft, custom sequence).

New canonical poly_def format
─────────────────────────────
{
  "polymer_name":  str,
  "polymer_type":  "homopolymer" | "copolymer",
  "sequence_type": "homopolymer" | "block" | "alternating" | "random"
                   | "periodic" | "graft" | "custom",
  "composition":   {"A": 0.5, "B": 0.5},        # optional, for random
  "monomer_types": {
      "A": {
          "name":           str,
          "monomer_atoms":  [{local_id, opls_id, charge, comment}, ...],
          "intra_bonds":    [[i,j], ...],
          "head_atom":      int,
          "tail_atom":      int,
          "n_atoms":        int,          # auto-filled if absent
      },
      "B": { ... },
  },
  "sequence": ["A","A","B","B",...],   # len == n_monomers

  # ── backward-compat fields (homopolymer only) ──
  "monomer_atoms": [...],
  "intra_bonds":   [...],
  "head_atom":     int,
  "tail_atom":     int,
}

Old format (single monomer_atoms) is auto-normalised on any entry point.
"""

from __future__ import annotations

from itertools import cycle
from typing import Any

import numpy as np

AVOGADRO  = 6.02214076e23
CM3_TO_A3 = 1.0e24


# ─────────────────────────────────────────────────────────────────────────────
# POLY_DEF NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_poly_def(poly_def: dict, n_monomers: int = 10) -> dict:
    """
    Convert any poly_def to the canonical multi-monomer format.
    Old format (single monomer_atoms) → wrapped as monomer_type 'A'.
    If sequence is absent or wrong length, it is (re-)generated.
    """
    poly_def = dict(poly_def)   # shallow copy — do not mutate caller's dict

    # ── already new format ───────────────────────────────────────────────────
    if "monomer_types" in poly_def:
        # ensure n_atoms
        for lbl, mdef in poly_def["monomer_types"].items():
            if "n_atoms" not in mdef:
                mdef["n_atoms"] = len(mdef.get("monomer_atoms", []))

        # ensure sequence of correct length
        labels = list(poly_def["monomer_types"].keys())
        seq = poly_def.get("sequence", [])
        if len(seq) != n_monomers:
            seq = generate_sequence(
                sequence_type   = poly_def.get("sequence_type", "homopolymer"),
                labels          = labels,
                n_monomers      = n_monomers,
                composition     = poly_def.get("composition"),
                custom_sequence = seq if poly_def.get("sequence_type") == "custom" else None,
                period          = poly_def.get("period"),
            )
            poly_def["sequence"] = [str(x) for x in seq]
        return poly_def

    # ── old format: wrap single monomer ──────────────────────────────────────
    if "monomer_atoms" not in poly_def:
        raise ValueError("poly_def must contain 'monomer_types' or 'monomer_atoms'.")

    mono_def = {
        "name":          poly_def.get("polymer_name", "monomer"),
        "monomer_atoms": poly_def["monomer_atoms"],
        "intra_bonds":   poly_def.get("intra_bonds", []),
        "head_atom":     poly_def["head_atom"],
        "tail_atom":     poly_def["tail_atom"],
        "n_atoms":       len(poly_def["monomer_atoms"]),
    }
    poly_def["monomer_types"]  = {"A": mono_def}
    poly_def["sequence"]       = ["A"] * n_monomers
    poly_def["polymer_type"]   = "homopolymer"
    poly_def["sequence_type"]  = "homopolymer"
    return poly_def


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_sequence(
    sequence_type:   str,
    labels:          list[str],
    n_monomers:      int,
    composition:     dict | None        = None,
    custom_sequence: list[str] | None   = None,
    period:          list[str] | None   = None,
    seed:            int | None         = None,
) -> list[str]:
    """
    Generate a monomer sequence.

    sequence_type options
    ─────────────────────
    'homopolymer' / single label → all same
    'block'        → equal contiguous blocks  [A…A B…B C…C]
    'alternating'  → A-B-A-B-…
    'random'       → random draw with optional composition weights
    'periodic'     → repeat a given period pattern
    'graft'        → main chain = labels[0], grafts at every 3rd position
    'custom'       → explicit list via custom_sequence
    """
    if not labels:
        raise ValueError("labels must be non-empty")

    if len(labels) == 1 or sequence_type in ("homopolymer",):
        return [labels[0]] * n_monomers

    if sequence_type == "custom" and custom_sequence is not None:
        seq = [s for s in custom_sequence if s in labels][:n_monomers]
        while len(seq) < n_monomers:
            seq.append(labels[-1])
        return seq

    if sequence_type == "block":
        n_types  = len(labels)
        per_type = n_monomers // n_types
        rem      = n_monomers % n_types
        seq: list[str] = []
        for i, lbl in enumerate(labels):
            seq.extend([lbl] * (per_type + (1 if i < rem else 0)))
        return seq

    if sequence_type == "alternating":
        c = cycle(labels)
        return [next(c) for _ in range(n_monomers)]

    if sequence_type == "random":
        rng = np.random.default_rng(seed)
        if composition:
            total = sum(composition.get(l, 0.0) for l in labels)
            probs = [composition.get(l, 0.0) / total for l in labels] if total else None
        else:
            probs = None
        return [str(x) for x in rng.choice(labels, size=n_monomers, p=probs)]

    if sequence_type == "periodic":
        pat = period or labels
        c   = cycle(pat)
        return [next(c) for _ in range(n_monomers)]

    if sequence_type == "graft":
        # main chain = labels[0], graft monomer = labels[1]
        main, graft = labels[0], labels[1]
        seq = []
        for i in range(n_monomers):
            seq.append(graft if (i > 0 and i % 3 == 0) else main)
        return seq

    # fallback → random
    return generate_sequence("random", labels, n_monomers, composition, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# OFFSET & MASS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_monomer_offsets(sequence: list[str], monomer_types: dict) -> list[int]:
    """
    Cumulative atom-count offsets for one chain.
    offsets[i]  = index of first atom of monomer i
    offsets[-1] = total atoms per chain
    """
    offsets = [0]
    for lbl in sequence:
        offsets.append(offsets[-1] + len(monomer_types[lbl]["monomer_atoms"]))
    return offsets


def compute_chain_mass(
    sequence: list[str],
    monomer_types: dict,
    opls_atoms: dict,
) -> float:
    """Total mass of one chain (g/mol)."""
    total = 0.0
    for lbl in sequence:
        for a in monomer_types[lbl]["monomer_atoms"]:
            oid = a["opls_id"]
            total += opls_atoms[oid]["mass"] if oid in opls_atoms else 12.011
    return total


def calculate_box_size_from_sequence(
    n_chains:      int,
    sequence:      list[str],
    monomer_types: dict,
    opls_atoms:    dict,
    density:       float = 0.9,
) -> float:
    """Return cubic box side length (Å) for a (co)polymer system."""
    chain_mass = compute_chain_mass(sequence, monomer_types, opls_atoms)
    total_mass = n_chains * chain_mass
    vol_cm3    = total_mass / (density * AVOGADRO)
    return (vol_cm3 * CM3_TO_A3) ** (1.0 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# CG SEQUENCE & TOPOLOGY DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def derive_cg_bead_sequence(
    sequence: list[str],
    bead_map_def: dict,
) -> list[int]:
    """
    Convert AA monomer sequence → ordered list of CG bead_type_ids for one chain.

    bead_types with 'monomer_type' key  → copolymer mapping
    bead_types without 'monomer_type'   → homopolymer (apply to every monomer)

    For monomers_per_bead > 1: the bead consumes that many consecutive monomers
    of the same label.  Mixed-type spans are not supported.
    """
    bead_types    = bead_map_def["bead_types"]
    has_mono_type = any("monomer_type" in bt for bt in bead_types)

    result: list[int] = []
    mono_idx = 0

    while mono_idx < len(sequence):
        label = sequence[mono_idx]

        if not has_mono_type:
            # Homopolymer path: every bead_type applies to this monomer
            for bt in bead_types:
                mpb = bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1
                result.append(bt["bead_type_id"])
                # monomers_per_bead handled by advancing mono_idx below
            # advance by max mpb
            max_mpb = max(
                bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1
                for bt in bead_types
            )
            mono_idx += max_mpb
        else:
            # Copolymer path: only bead_types whose monomer_type matches
            matched = [bt for bt in bead_types if bt.get("monomer_type") == label]
            if not matched:
                # Fallback: bead_types without monomer_type field
                matched = [bt for bt in bead_types if "monomer_type" not in bt]
            for bt in matched:
                result.append(bt["bead_type_id"])
            # advance by max mpb of matched beads
            max_mpb = max(
                (bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1)
                for bt in matched
            ) if matched else 1
            mono_idx += max_mpb

    return result


def derive_cg_topology(
    cg_bead_sequence: list[int],
) -> dict:
    """
    Derive all CG bond, angle, and LJ pair types from an actual bead sequence.

    Uses only types that ACTUALLY appear in the sequence (no phantom types).
    Bond and angle types reflect the real connectivity order (not sorted pairs)
    to preserve direction information (e.g. [1,1,2] ≠ [2,1,1]).
    LJ pairs: every unique unordered pair (including self-pairs).
    """
    unique_ids = sorted(set(cg_bead_sequence))

    # Bond types: consecutive pairs (sorted for canonical form)
    bond_pairs: list[list[int]] = []
    seen_bonds: set[tuple] = set()
    for i in range(len(cg_bead_sequence) - 1):
        key = tuple(sorted([cg_bead_sequence[i], cg_bead_sequence[i + 1]]))
        if key not in seen_bonds:
            seen_bonds.add(key)
            bond_pairs.append(list(key))

    # Angle types: consecutive triplets  (keep direction!)
    angle_triplets: list[list[int]] = []
    seen_angles: set[tuple] = set()
    for i in range(len(cg_bead_sequence) - 2):
        tri = (cg_bead_sequence[i], cg_bead_sequence[i + 1], cg_bead_sequence[i + 2])
        rev = tri[::-1]
        key = min(tri, rev)
        if key not in seen_angles:
            seen_angles.add(key)
            angle_triplets.append(list(tri))

    # LJ pairs: all unique unordered combinations
    lj_pairs: list[list[int]] = []
    for i in range(len(unique_ids)):
        for j in range(i, len(unique_ids)):
            lj_pairs.append([unique_ids[i], unique_ids[j]])

    # Guarantee BB-BB-BB backbone angle if 3+ of the same type
    backbone_id = cg_bead_sequence[0] if cg_bead_sequence else 1
    bb_angle    = [backbone_id, backbone_id, backbone_id]
    if bb_angle not in angle_triplets:
        angle_triplets.insert(0, bb_angle)

    return {
        "bond_type_pairs":     bond_pairs,
        "angle_type_triplets": angle_triplets,
        "lj_type_pairs":       lj_pairs,
        "cg_bead_sequence":    cg_bead_sequence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE DISPLAY HELPERS (for user messages)
# ─────────────────────────────────────────────────────────────────────────────

def sequence_summary(
    sequence: list[str],
    monomer_types: dict,
) -> str:
    """
    Human-readable sequence summary, e.g.
    "block: A×5 - B×5" or "A-B-A-B-A-B-A-B-A-B"
    """
    if not sequence:
        return "(empty)"

    # Detect run-length encoding
    runs: list[tuple[str, int]] = []
    cur_lbl = sequence[0]
    cur_cnt = 1
    for lbl in sequence[1:]:
        if lbl == cur_lbl:
            cur_cnt += 1
        else:
            runs.append((cur_lbl, cur_cnt))
            cur_lbl, cur_cnt = lbl, 1
    runs.append((cur_lbl, cur_cnt))

    # Short sequence: show full
    if len(sequence) <= 12:
        return "-".join(sequence)

    # Long: show runs
    parts = []
    for lbl, cnt in runs:
        name = monomer_types.get(lbl, {}).get("name", lbl)
        parts.append(f"{name}×{cnt}" if cnt > 1 else name)
    return " - ".join(parts)
