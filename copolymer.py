"""
copolymer.py
============
Multi-CRU copolymer support for CGMas.

Architectures
─────────────
homopolymer : single CRU  (단순 위임 — 기존 aa_constructor 그대로)
random      : A/B/... 랜덤 배열, fraction 지정
alternating : ABABAB...
block       : AAAA...BBBB... (block_lengths 또는 equal split)
periodic    : 사용자 지정 패턴, e.g. [0,0,1] → AABAABAAB...
graft       : backbone(CRU 0) + side chain(CRU 1), 일정 간격으로 접합

Usage (노트북 aa_constructor_node에서 호출):
    from copolymer import build_and_write_copoly, CopolymerSpec

    spec = CopolymerSpec(architecture="random", fractions=[0.5, 0.5])
    system = build_and_write_copoly(
        poly_defs   = [poly_def_A, poly_def_B],
        opls_dir    = OPLS_DIR,
        output_path = datafile_path,
        spec        = spec,
        n_chains    = 20,
        n_monomers  = 20,
        density     = 0.9,
        seed        = 42,
    )
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from aa_constructor import (
    load_opls,
    calculate_box_size,
    place_monomer,
    find_rings,
    enumerate_angles,
    enumerate_dihedrals,
    lookup_bond,
    lookup_angle,
    lookup_dihedral,
    write_lammps_datafile,   # 재사용
    _SpatialGrid,
    AVOGADRO,
    CM3_TO_A3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Copolymer specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CopolymerSpec:
    """
    Describes how multiple CRUs are arranged along a chain.

    Attributes
    ----------
    architecture  : sequence topology
    fractions     : mole fractions of each CRU type (sums to 1).
                    For alternating/block/periodic, used only for box size.
    block_lengths : for "block" — list of integers, e.g. [5, 5] for diblock.
                    Repeated as needed to fill n_monomers.
    pattern       : for "periodic" — CRU index pattern, e.g. [0, 0, 1].
    graft_interval: for "graft" — attach branch every N backbone monomers.
    graft_length  : for "graft" — number of branch monomers.
    graft_cru_idx : for "graft" — which poly_defs index is the branch (default 1).
    branch_local_id: for "graft" — local_id in backbone CRU where branch attaches.
    seed          : random seed for sequence generation.
    """
    architecture:   Literal["homopolymer","random","alternating","block","periodic","graft"] = "homopolymer"
    fractions:      list[float] = field(default_factory=lambda: [1.0])
    block_lengths:  list[int]   = field(default_factory=list)
    pattern:        list[int]   = field(default_factory=list)
    graft_interval: int  = 4
    graft_length:   int  = 3
    graft_cru_idx:  int  = 1
    branch_local_id: int | None = None   # backbone atom where branch bonds


# ─────────────────────────────────────────────────────────────────────────────
# Sequence generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_sequence(n_monomers: int, spec: CopolymerSpec,
                      rng: np.random.Generator) -> list[int]:
    """
    Return a list of CRU indices (0-based) of length n_monomers.
    For graft, returns the backbone sequence only (branches handled separately).
    """
    arch    = spec.architecture
    n_types = len(spec.fractions)

    if arch == "homopolymer" or n_types == 1:
        return [0] * n_monomers

    if arch == "random":
        raw   = spec.fractions if spec.fractions else [1.0/n_types]*n_types
        fracs = np.array(raw[:n_types], dtype=float)
        fracs /= fracs.sum()
        return list(rng.choice(n_types, size=n_monomers, p=fracs))

    if arch == "alternating":
        return [i % n_types for i in range(n_monomers)]

    if arch == "block":
        if spec.block_lengths:
            unit = []
            for cru_idx, bl in enumerate(spec.block_lengths):
                unit.extend([cru_idx % n_types] * bl)
        else:
            bl = max(1, n_monomers // n_types)
            unit = []
            for t in range(n_types):
                unit.extend([t] * bl)
        # tile to fill n_monomers
        reps   = (n_monomers + len(unit) - 1) // len(unit)
        tiled  = (unit * reps)[:n_monomers]
        return tiled

    if arch == "periodic":
        if not spec.pattern:
            return [0] * n_monomers
        return [spec.pattern[i % len(spec.pattern)] for i in range(n_monomers)]

    if arch == "graft":
        return [0] * n_monomers   # backbone only

    return [0] * n_monomers


def graft_branch_positions(n_monomers: int, spec: CopolymerSpec) -> list[int]:
    """Return backbone monomer indices that carry a branch."""
    return list(range(spec.graft_interval - 1, n_monomers, spec.graft_interval))


# ─────────────────────────────────────────────────────────────────────────────
# Per-CRU local info builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_local_info(poly_def: dict, opls_atoms: dict) -> dict[int, dict]:
    li = {}
    for a in poly_def["monomer_atoms"]:
        lid  = a["local_id"]
        oid  = a["opls_id"]
        data = opls_atoms[oid].copy()
        if "charge" in a and a["charge"] != 0.0:
            data["charge"] = float(a["charge"])
        li[lid] = data
    return li


# ─────────────────────────────────────────────────────────────────────────────
# Unified atom-type registry
# ─────────────────────────────────────────────────────────────────────────────

def _unified_type_registry(poly_defs: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Collect all distinct opls_ids across all CRU definitions."""
    seen: list[str] = []
    for pdef in poly_defs:
        for a in pdef["monomer_atoms"]:
            if a["opls_id"] not in seen:
                seen.append(a["opls_id"])
    return seen, {oid: i + 1 for i, oid in enumerate(seen)}


# ─────────────────────────────────────────────────────────────────────────────
# Box size from average monomer mass
# ─────────────────────────────────────────────────────────────────────────────

def _avg_monomer_mass(poly_defs: list[dict], all_li: list[dict], spec: CopolymerSpec) -> float:
    masses = [sum(li[a["local_id"]]["mass"] for a in pdef["monomer_atoms"])
              for pdef, li in zip(poly_defs, all_li)]
    raw = spec.fractions[:len(masses)] if spec.fractions else []
    if not raw:
        return float(np.mean(masses)) if masses else 0.0
    fracs = np.array(raw, dtype=float)
    fracs /= fracs.sum()
    return float(np.dot(fracs, masses[:len(fracs)]))


# ─────────────────────────────────────────────────────────────────────────────
# Core multi-CRU system builder
# ─────────────────────────────────────────────────────────────────────────────

def build_polymer_system_multi(
    poly_defs:   list[dict],
    opls:        dict,
    spec:        CopolymerSpec,
    n_chains:    int   = 20,
    n_monomers:  int   = 10,
    density:     float = 0.9,
    seed:        int | None = None,
) -> dict:
    """
    Build a multi-CRU amorphous polymer system.

    Returns the same system dict as aa_constructor.build_polymer_system,
    extended with 'copolymer_spec' and 'chain_sequences' fields.
    """
    rng = np.random.default_rng(seed)
    opls_atoms = opls["atom"]

    # ── per-CRU info ────────────────────────────────────────────────────────
    all_li = [_build_local_info(pd, opls_atoms) for pd in poly_defs]

    # ── unified type registry ───────────────────────────────────────────────
    seen_types, opls_id_to_type_id = _unified_type_registry(poly_defs)

    # ── box size ────────────────────────────────────────────────────────────
    avg_mass = _avg_monomer_mass(poly_defs, all_li, spec)
    # For graft: account for branch monomers
    if spec.architecture == "graft":
        n_branch_per_chain = len(graft_branch_positions(n_monomers, spec)) * spec.graft_length
        branch_mass = (sum(all_li[spec.graft_cru_idx][a["local_id"]]["mass"]
                           for a in poly_defs[spec.graft_cru_idx]["monomer_atoms"])
                       if len(poly_defs) > spec.graft_cru_idx else 0.0)
        avg_mass = (avg_mass * n_monomers + branch_mass * n_branch_per_chain) / (n_monomers + n_branch_per_chain)

    box_L = calculate_box_size(n_chains, n_monomers, avg_mass, density)

    # ── spatial clash grid ──────────────────────────────────────────────────
    clash_grid = _SpatialGrid(cell_size=3.0)

    all_atoms:     list = []
    all_bonds:     list = []
    all_angles:    list = []
    all_dihedrals: list = []

    # Per-chain sequence storage
    chain_sequences: list[list[int]] = []

    # ── grid-based origin placement ─────────────────────────────────────────
    grid_n   = int(np.ceil(n_chains ** (1.0 / 3.0)))
    cell_sz  = box_L / grid_n
    grid_pts = [(i, j, k)
                for i in range(grid_n)
                for j in range(grid_n)
                for k in range(grid_n)]
    rng.shuffle(grid_pts)

    # Global atom-ID counter (needed because chains have variable atom counts)
    gid_counter = [1]

    def next_gid():
        g = gid_counter[0]
        gid_counter[0] += 1
        return g

    MAX_CHAIN_RETRIES = 5

    for chain_i in range(n_chains):
        mol_id = chain_i + 1

        # ── generate backbone sequence ──────────────────────────────────────
        seq   = generate_sequence(n_monomers, spec, rng)
        chain_sequences.append(seq)

        # ── graft branch positions ─────────────────────────────────────────
        branch_positions = (graft_branch_positions(n_monomers, spec)
                            if spec.architecture == "graft" else [])

        gi, gj, gk = grid_pts[chain_i % len(grid_pts)]
        origin     = (np.array([gi, gj, gk], dtype=float) + 0.5) * cell_sz

        accepted_atoms:     list = []
        accepted_positions: list[np.ndarray] = []

        for retry in range(MAX_CHAIN_RETRIES + 1):
            jit = min(0.20 + 0.15 * retry, 0.50)
            o   = origin + rng.uniform(-jit * cell_sz, jit * cell_sz, size=3)
            o   = np.clip(o, 0.0, box_L)

            anchor_pos:    np.ndarray       = o.copy()
            prev_tail_pos: np.ndarray | None = None

            chain_atoms_tmp:     list = []
            chain_positions_tmp: list[np.ndarray] = []
            total_fallbacks = 0

            # monomer-global-id map: (mono_i, local_id) → gid  (for bonds)
            mono_gid: dict[tuple[int,int], int] = {}
            # For branch monomers: (branch_mono_i, local_id) → gid
            branch_gid: dict[tuple[int,int,int], int] = {}
            # Track last tail position & atom (for branch attachment)
            backbone_tail_pos:  dict[int, np.ndarray] = {}
            backbone_tail_gids: dict[int, int]        = {}
            backbone_branch_atom_pos: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            cur_anchor    = anchor_pos.copy()
            cur_prev_tail = prev_tail_pos

            for mono_i, cru_idx in enumerate(seq):
                pdef  = poly_defs[cru_idx]
                li    = all_li[cru_idx]
                intra = [list(b) for b in pdef["intra_bonds"]]
                head  = pdef["head_atom"]
                tail  = pdef["tail_atom"]

                local_pos, n_fb = place_monomer(
                    intra_bonds   = intra,
                    local_info    = li,
                    opls          = opls,
                    head_lid      = head,
                    anchor_pos    = cur_anchor,
                    prev_tail_pos = cur_prev_tail,
                    rng           = rng,
                    grid          = clash_grid,
                )
                total_fallbacks += n_fb

                for lid, pos in local_pos.items():
                    gid = next_gid()
                    mono_gid[(mono_i, lid)] = gid
                    info = li[lid]
                    chain_atoms_tmp.append((gid, mol_id,
                                            opls_id_to_type_id[info["opls_id"]],
                                            info["charge"],
                                            pos[0], pos[1], pos[2]))
                    chain_positions_tmp.append(pos)

                tail_pos = local_pos[tail]
                head_pos = local_pos[head]
                backbone_tail_pos[mono_i]  = tail_pos
                backbone_tail_gids[mono_i] = mono_gid[(mono_i, tail)]

                # Store branch attachment atom position if needed
                if spec.architecture == "graft" and mono_i in branch_positions:
                    branch_lid = spec.branch_local_id or tail
                    backbone_branch_atom_pos[mono_i] = (
                        local_pos.get(branch_lid, tail_pos),
                        tail_pos,   # "grandparent" for branch z-matrix
                    )

                # Advance anchor for next backbone monomer
                bt_t = li[tail]["bond_type"]
                bt_h = all_li[seq[mono_i + 1] if mono_i + 1 < len(seq) else cru_idx][
                    poly_defs[seq[mono_i + 1] if mono_i + 1 < len(seq) else cru_idx]["head_atom"]
                ]["bond_type"]
                inter_bl = lookup_bond(opls, bt_t, bt_h)["r0"]
                grow_dir = tail_pos - head_pos
                norm = np.linalg.norm(grow_dir)
                grow_dir = grow_dir / norm if norm > 1e-8 else rng.normal(size=3) / 1.0
                cur_anchor    = tail_pos + inter_bl * grow_dir
                cur_prev_tail = tail_pos

            # ── graft branches ──────────────────────────────────────────────
            if spec.architecture == "graft" and len(poly_defs) > spec.graft_cru_idx:
                br_pdef = poly_defs[spec.graft_cru_idx]
                br_li   = all_li[spec.graft_cru_idx]
                br_intra= [list(b) for b in br_pdef["intra_bonds"]]
                br_head = br_pdef["head_atom"]
                br_tail = br_pdef["tail_atom"]

                for bb_mono_i in branch_positions:
                    if bb_mono_i not in backbone_branch_atom_pos:
                        continue
                    br_anchor, br_gp = backbone_branch_atom_pos[bb_mono_i]

                    br_cur_anchor    = br_anchor
                    br_cur_prev_tail = br_gp

                    for br_mono in range(spec.graft_length):
                        local_pos, n_fb = place_monomer(
                            intra_bonds   = br_intra,
                            local_info    = br_li,
                            opls          = opls,
                            head_lid      = br_head,
                            anchor_pos    = br_cur_anchor,
                            prev_tail_pos = br_cur_prev_tail,
                            rng           = rng,
                            grid          = clash_grid,
                        )
                        total_fallbacks += n_fb

                        for lid, pos in local_pos.items():
                            gid = next_gid()
                            branch_gid[(bb_mono_i, br_mono, lid)] = gid
                            info = br_li[lid]
                            chain_atoms_tmp.append((gid, mol_id,
                                                    opls_id_to_type_id[info["opls_id"]],
                                                    info["charge"],
                                                    pos[0], pos[1], pos[2]))
                            chain_positions_tmp.append(pos)

                        tail_pos = local_pos[br_tail]
                        head_pos = local_pos[br_head]
                        bt_t = br_li[br_tail]["bond_type"]
                        inter_bl = lookup_bond(opls, bt_t, bt_t)["r0"]
                        grow_dir = tail_pos - head_pos
                        norm = np.linalg.norm(grow_dir)
                        grow_dir = grow_dir / norm if norm > 1e-8 else rng.normal(size=3) / 1.0
                        br_cur_anchor    = tail_pos + inter_bl * grow_dir
                        br_cur_prev_tail = tail_pos

            _FALLBACK_LIMIT = max(1, (len(chain_atoms_tmp)) // 5)
            if total_fallbacks <= _FALLBACK_LIMIT or retry == MAX_CHAIN_RETRIES:
                accepted_atoms     = chain_atoms_tmp
                accepted_positions = chain_positions_tmp
                break
            else:
                warnings.warn(f"Chain {mol_id}: {total_fallbacks} fallbacks, retry {retry+1}")
                # Reset gid_counter to reclaim IDs from this failed attempt
                gid_counter[0] -= len(chain_atoms_tmp)

        # Commit
        for atom in accepted_atoms:
            all_atoms.append(atom)
        for pos in accepted_positions:
            clash_grid.add(pos)

        # ── backbone bonds ──────────────────────────────────────────────────
        for mono_i, cru_idx in enumerate(seq):
            pdef  = poly_defs[cru_idx]
            intra = [list(b) for b in pdef["intra_bonds"]]
            for (li_i, li_j) in intra:
                all_bonds.append((mono_gid[(mono_i, li_i)],
                                  mono_gid[(mono_i, li_j)]))
            # Inter-monomer backbone bond
            if mono_i < len(seq) - 1:
                all_bonds.append((mono_gid[(mono_i,     pdef["tail_atom"])],
                                  mono_gid[(mono_i + 1, poly_defs[seq[mono_i+1]]["head_atom"])]))

        # ── graft branch bonds ──────────────────────────────────────────────
        if spec.architecture == "graft" and len(poly_defs) > spec.graft_cru_idx:
            br_pdef = poly_defs[spec.graft_cru_idx]
            br_intra= [list(b) for b in br_pdef["intra_bonds"]]
            for bb_mono_i in branch_positions:
                for br_mono in range(spec.graft_length):
                    for (li_i, li_j) in br_intra:
                        if (bb_mono_i, br_mono, li_i) in branch_gid:
                            all_bonds.append((branch_gid[(bb_mono_i, br_mono, li_i)],
                                              branch_gid[(bb_mono_i, br_mono, li_j)]))
                    # Branch–backbone bond (first branch monomer only)
                    if br_mono == 0:
                        branch_lid = spec.branch_local_id or poly_defs[0]["tail_atom"]
                        if (bb_mono_i, branch_lid) in mono_gid and (bb_mono_i, 0, br_pdef["head_atom"]) in branch_gid:
                            all_bonds.append((mono_gid[(bb_mono_i, branch_lid)],
                                              branch_gid[(bb_mono_i, 0, br_pdef["head_atom"])]))
                    # Inter-branch monomer bonds
                    elif (bb_mono_i, br_mono-1, br_pdef["tail_atom"]) in branch_gid:
                        all_bonds.append((branch_gid[(bb_mono_i, br_mono-1, br_pdef["tail_atom"])],
                                          branch_gid[(bb_mono_i, br_mono,   br_pdef["head_atom"])]))

        # ── angles & dihedrals ──────────────────────────────────────────────
        n_new = len(all_bonds) - len(all_angles)   # approximate
        chain_bond_slice = all_bonds[-(len(all_bonds) - sum(
            len(poly_defs[seq[m]]["intra_bonds"]) for m in range(len(seq))
        ) - (len(seq) - 1)):]
        # Simpler: enumerate from all bonds belonging to this mol_id
        this_chain_bonds = [(ai, aj) for tid, ai, aj in
                             [(0, ai, aj) for (ai, aj) in all_bonds]
                             if ai in {a[0] for a in accepted_atoms}]
        # Use accepted_atoms gids for filtering
        chain_gids = {a[0] for a in accepted_atoms}
        this_chain_bonds_clean = [(ai, aj) for (ai, aj) in all_bonds
                                  if ai in chain_gids or aj in chain_gids]
        for triple in enumerate_angles(this_chain_bonds_clean):
            all_angles.append(triple)
        for quad in enumerate_dihedrals(this_chain_bonds_clean):
            all_dihedrals.append(quad)

    # ── assign FF types ─────────────────────────────────────────────────────
    gid_to_bt: dict[int, str] = {}
    for (gid, mol_id, tid, charge, x, y, z) in all_atoms:
        gid_to_bt[gid] = opls_atoms[seen_types[tid - 1]]["bond_type"]

    def _type_id(keys, tmap, key):
        rk = key[::-1]
        if key in tmap: return tmap[key]
        if rk  in tmap: return tmap[rk]
        tid = len(keys) + 1
        tmap[key] = tid
        keys.append(key)
        return tid

    bond_type_keys: list = [];  bond_type_map: dict = {}
    typed_bonds = []
    for (ai, aj) in all_bonds:
        if ai not in gid_to_bt or aj not in gid_to_bt:
            continue
        key = (gid_to_bt[ai], gid_to_bt[aj])
        typed_bonds.append((_type_id(bond_type_keys, bond_type_map, key), ai, aj))

    angle_type_keys: list = [];  angle_type_map: dict = {}
    typed_angles = []
    for (ai, aj, ak) in all_angles:
        if not all(g in gid_to_bt for g in (ai, aj, ak)): continue
        key = (gid_to_bt[ai], gid_to_bt[aj], gid_to_bt[ak])
        typed_angles.append((_type_id(angle_type_keys, angle_type_map, key), ai, aj, ak))

    dih_type_keys: list = [];  dih_type_map: dict = {}
    typed_dihedrals = []
    for (ai, aj, ak, al) in all_dihedrals:
        if not all(g in gid_to_bt for g in (ai, aj, ak, al)): continue
        key = (gid_to_bt[ai], gid_to_bt[aj], gid_to_bt[ak], gid_to_bt[al])
        typed_dihedrals.append((_type_id(dih_type_keys, dih_type_map, key), ai, aj, ak, al))

    return {
        "poly_def":         poly_defs[0],    # primary (for write_lammps_datafile compat)
        "poly_defs":        poly_defs,
        "copolymer_spec":   spec,
        "chain_sequences":  chain_sequences,
        "n_chains":         n_chains,
        "n_monomers":       n_monomers,
        "density":          density,
        "box_L":            box_L,
        "monomer_mass":     avg_mass,
        "atom_type_ids":    opls_id_to_type_id,
        "seen_types":       seen_types,
        "bond_type_keys":   bond_type_keys,
        "angle_type_keys":  angle_type_keys,
        "dih_type_keys":    dih_type_keys,
        "atoms":            all_atoms,
        "bonds":            typed_bonds,
        "angles":           typed_angles,
        "dihedrals":        typed_dihedrals,
        "gid_to_bt":        gid_to_bt,
        "opls_atoms":       opls_atoms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_and_write_copoly(
    poly_defs:   list[dict] | list[str],
    opls_dir:    str,
    output_path: str,
    spec:        CopolymerSpec | None = None,
    n_chains:    int   = 20,
    n_monomers:  int   = 10,
    density:     float = 0.9,
    seed:        int | None = None,
) -> dict:
    """
    High-level entry point for copolymer data file generation.

    poly_defs can be a list of dicts or JSON strings.
    If spec is None, defaults to homopolymer (poly_defs[0] only).
    """
    poly_defs = [json.loads(pd) if isinstance(pd, str) else pd for pd in poly_defs]
    if spec is None:
        spec = CopolymerSpec(architecture="homopolymer", fractions=[1.0])

    opls   = load_opls(opls_dir)
    system = build_polymer_system_multi(
        poly_defs   = poly_defs,
        opls        = opls,
        spec        = spec,
        n_chains    = n_chains,
        n_monomers  = n_monomers,
        density     = density,
        seed        = seed,
    )
    write_lammps_datafile(system, opls, output_path)

    n_types = len(poly_defs)
    names   = [pd.get("polymer_name", f"CRU{i}") for i, pd in enumerate(poly_defs)]
    print(f"\n✅  Copolymer data file: {output_path}")
    print(f"   Architecture : {spec.architecture}")
    print(f"   Components   : {names}")
    if n_types > 1:
        fracs = np.array(spec.fractions[:n_types])
        fracs /= fracs.sum()
        print(f"   Fractions    : {[round(f,3) for f in fracs]}")
    if spec.architecture == "graft":
        print(f"   Graft interval: every {spec.graft_interval} backbone monomers")
        print(f"   Branch length : {spec.graft_length} monomers")
    print(f"   Chains       : {n_chains} × {n_monomers} monomers")
    print(f"   Avg monomer M: {system['monomer_mass']:.3f} g/mol")
    print(f"   Box size     : {system['box_L']:.4f} Å  (density {density} g/cm³)")
    print(f"   Atoms        : {len(system['atoms'])}  Bonds: {len(system['bonds'])}  "
          f"Angles: {len(system['angles'])}  Dihedrals: {len(system['dihedrals'])}")
    return system
