"""
mapper.py
=========
Bead mapping tool: reads equilibrated AA LAMMPS data file,
applies user-defined bead mapping, computes center-of-mass positions,
and writes a CG LAMMPS data file (atom_style molecular).

Bead mapping definition JSON (produced by LLM):
{
  "bead_types": [
    {
      "bead_type_id":    1,
      "name":            "BB",
      "description":     "backbone CH2-CH",
      "atom_local_ids":  [1, 2, 3, 4, 5],
      "monomers_per_bead": 1
    }
  ]
}
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AAAtom:
    aid:   int
    mol:   int
    atype: int
    q:     float
    x:     float
    y:     float
    z:     float


@dataclass
class Box:
    xlo: float; xhi: float
    ylo: float; yhi: float
    zlo: float; zhi: float

    @property
    def L(self) -> np.ndarray:
        return np.array([self.xhi - self.xlo,
                         self.yhi - self.ylo,
                         self.zhi - self.zlo])


# ── Read AA data ──────────────────────────────────────────────────────────────

def read_aa_data(data_path: str):
    atoms:  dict[int, AAAtom] = {}
    bonds:  list[tuple]       = []
    masses: dict[int, float]  = {}
    box = Box(0, 0, 0, 0, 0, 0)
    section = None

    with open(data_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue

            m = re.match(r'([\-\d.eE+]+)\s+([\-\d.eE+]+)\s+xlo xhi', s)
            if m: box.xlo, box.xhi = float(m[1]), float(m[2]); continue
            m = re.match(r'([\-\d.eE+]+)\s+([\-\d.eE+]+)\s+ylo yhi', s)
            if m: box.ylo, box.yhi = float(m[1]), float(m[2]); continue
            m = re.match(r'([\-\d.eE+]+)\s+([\-\d.eE+]+)\s+zlo zhi', s)
            if m: box.zlo, box.zhi = float(m[1]), float(m[2]); continue

            if s.startswith('Masses'):   section = 'masses'; continue
            if s.startswith('Atoms'):    section = 'atoms';  continue
            if s.startswith('Bonds'):    section = 'bonds';  continue
            if re.match(r'(Angles|Dihedrals|Pair|Bond Coeffs|Angle|Dihedral)', s):
                section = 'skip'; continue

            if section == 'masses':
                p = s.split()
                if len(p) >= 2:
                    try: masses[int(p[0])] = float(p[1])
                    except ValueError: pass

            elif section == 'atoms':
                p = s.split()
                if len(p) >= 7:
                    try:
                        atoms[int(p[0])] = AAAtom(
                            int(p[0]), int(p[1]), int(p[2]),
                            float(p[3]), float(p[4]), float(p[5]), float(p[6])
                        )
                    except ValueError: pass

            elif section == 'bonds':
                p = s.split()
                if len(p) >= 4:
                    try: bonds.append((int(p[2]), int(p[3])))
                    except ValueError: pass

    return atoms, bonds, box, masses


# ── Topology helpers ──────────────────────────────────────────────────────────

def build_mol_adj(atoms, bonds):
    mol_atoms = defaultdict(list)
    for aid, a in atoms.items():
        mol_atoms[a.mol].append(aid)

    adj = defaultdict(set)
    for a1, a2 in bonds:
        adj[a1].add(a2)
        adj[a2].add(a1)

    return dict(mol_atoms), adj


def linear_chain_order(mol_atom_ids: list, adj: dict) -> list[int]:
    """
    Return the atom ordering used to slice a molecule into per-monomer groups.

    The AA builder writes atoms monomer-by-monomer, and the LLM's
    `atom_local_ids` are defined against that monomer-internal ordering, so the
    *atom-id order* is the ground truth for slicing — NOT a chemical-bond walk.

    The previous implementation walked the graph from an arbitrary degree-1
    atom. For branched polymers (poly(4-methyl-1-pentene), poly(1-butene),
    poly(α-methylstyrene), …) the molecule has many degree-1 atoms (every
    hydrogen and every side-chain tip), so the walk started on a hydrogen or
    wandered into a side chain, scrambling the per-monomer slices. That made
    side-chain beads look like in-chain beads (e.g. an A-B-A-B alternating
    chain instead of an A backbone with B pendants), which corrupted the
    bond/angle topology and the density.

    Strategy:
      • If the molecule is a simple unbranched path (every atom has degree ≤ 2
        in the molecular subgraph), a bond walk is unambiguous and we use it.
      • Otherwise (any branching — i.e. real side chains or hydrogens present)
        we trust the atom-id order, which matches how the data file was built.
    """
    id_order = sorted(mol_atom_ids)
    id_set   = set(mol_atom_ids)
    sub = {a: [n for n in adj[a] if n in id_set] for a in mol_atom_ids}

    # Branching or any degree>2 → atom-id order is the reliable slicer.
    if any(len(nbrs) > 2 for nbrs in sub.values()):
        return id_order

    deg1 = [a for a, nbrs in sub.items() if len(nbrs) == 1]
    if len(deg1) != 2:
        # rings / fragments / isolated atoms → fall back to id order
        return id_order

    start = min(deg1)          # deterministic endpoint
    order = [start]
    prev, cur = None, start
    while True:
        nbrs = [n for n in sub[cur] if n != prev]
        if not nbrs:
            break
        order.append(nbrs[0])
        prev, cur = cur, nbrs[0]

    return order if len(order) == len(mol_atom_ids) else id_order


# ── Center of mass with PBC ───────────────────────────────────────────────────

def com_pbc(positions: np.ndarray, masses: np.ndarray, box_L: np.ndarray) -> np.ndarray:
    ref = positions[0].copy()
    unwrapped = positions.copy()
    for i in range(1, len(positions)):
        dr = positions[i] - ref
        dr -= box_L * np.round(dr / box_L)
        unwrapped[i] = ref + dr
    return np.average(unwrapped, axis=0, weights=masses)


# ── Bead mapping ──────────────────────────────────────────────────────────────

def apply_bead_mapping(atoms, bonds, box, masses_map, poly_def, bead_map_def, n_chains, n_monomers):
    from polymer import normalize_poly_def, compute_monomer_offsets

    mol_atoms, adj = build_mol_adj(atoms, bonds)
    box_L      = box.L
    bead_types = bead_map_def["bead_types"]

    if poly_def is None:
        raise ValueError("poly_def이 없습니다.")
    poly_def_n    = normalize_poly_def(poly_def, n_monomers)
    sequence      = poly_def_n["sequence"]
    monomer_types = poly_def_n["monomer_types"]
    has_mono_type = any("monomer_type" in bt for bt in bead_types)
    avg_mass = sum(masses_map.values()) / max(len(masses_map), 1) if masses_map else 12.011

    cg_atoms = []
    bead_id  = 1

    for mid in sorted(mol_atoms.keys()):
        atom_order    = linear_chain_order(mol_atoms[mid], adj)
        n_chain_atoms = len(atom_order)
        offsets       = compute_monomer_offsets(sequence, monomer_types)

        monomer_groups = []
        for mi, lbl in enumerate(sequence):
            s, e = offsets[mi], offsets[mi + 1]
            if e <= n_chain_atoms:
                monomer_groups.append((lbl, atom_order[s:e]))

        max_mpb  = max((bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1) for bt in bead_types)
        mono_idx = 0

        while mono_idx < len(monomer_groups):
            cur_label = monomer_groups[mono_idx][0]
            applicable = ([bt for bt in bead_types if bt.get("monomer_type") == cur_label]
                         or [bt for bt in bead_types if "monomer_type" not in bt]) if has_mono_type else bead_types

            for bt in applicable:
                mpb   = bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1
                lids  = bt.get("atom_local_ids") or bt.get("local_ids_in_monomer", [])
                bt_id = bt["bead_type_id"]
                bead_atom_ids = []
                for m_off in range(mpb):
                    mi = mono_idx + m_off
                    if mi >= len(monomer_groups): break
                    _, mg = monomer_groups[mi]
                    for lid in lids:
                        idx = lid - 1
                        if 0 <= idx < len(mg):
                            bead_atom_ids.append(mg[idx])
                if not bead_atom_ids:
                    continue
                pos  = np.array([[atoms[a].x, atoms[a].y, atoms[a].z] for a in bead_atom_ids])
                ms   = np.array([masses_map.get(atoms[a].atype, avg_mass) for a in bead_atom_ids])
                com  = com_pbc(pos, ms, box_L)
                cg_atoms.append({"bead_id": bead_id, "mol": mid, "bead_type": bt_id,
                                  "mass": float(np.sum(ms)), "x": com[0], "y": com[1], "z": com[2]})
                bead_id += 1
            mono_idx += max_mpb

    # Rebuild bead sequence for topology
    bead_seq_in_mol = defaultdict(list)
    bead_atoms_of   = {}   # bead_id -> set of AA atom ids it contains
    bid_cursor = 1
    for mid in sorted(mol_atoms.keys()):
        atom_order    = linear_chain_order(mol_atoms[mid], adj)
        n_chain_atoms = len(atom_order)
        offsets       = compute_monomer_offsets(sequence, monomer_types)
        monomer_groups = []
        for mi, lbl in enumerate(sequence):
            s, e = offsets[mi], offsets[mi+1]
            if e <= n_chain_atoms:
                monomer_groups.append((lbl, atom_order[s:e]))
        max_mpb  = max((bt.get("monomers_per_bead") or bt.get("repeating_monomers",1) or 1) for bt in bead_types)
        mono_idx = 0
        while mono_idx < len(monomer_groups):
            cur_label = monomer_groups[mono_idx][0]
            applicable = ([bt for bt in bead_types if bt.get("monomer_type") == cur_label]
                         or [bt for bt in bead_types if "monomer_type" not in bt]) if has_mono_type else bead_types
            for bt in applicable:
                mpb  = bt.get("monomers_per_bead") or bt.get("repeating_monomers",1) or 1
                lids = bt.get("atom_local_ids") or bt.get("local_ids_in_monomer", [])
                bead_atom_ids = []
                for m_off in range(mpb):
                    mi = mono_idx + m_off
                    if mi >= len(monomer_groups): break
                    _, mg = monomer_groups[mi]
                    for lid in lids:
                        idx = lid - 1
                        if 0 <= idx < len(mg):
                            bead_atom_ids.append(mg[idx])
                if not bead_atom_ids:
                    continue
                bead_seq_in_mol[mid].append((mono_idx, bt["bead_type_id"], bid_cursor))
                bead_atoms_of[bid_cursor] = set(bead_atom_ids)
                bid_cursor += 1
            mono_idx += max_mpb

    backbone_bt_id = next((bt["bead_type_id"] for bt in bead_types if bt.get("is_backbone", False)),
                          bead_types[0]["bead_type_id"])
    n_bead_types   = len(bead_types)

    mol_bead_map = defaultdict(list)
    for ba in cg_atoms:
        mol_bead_map[ba["mol"]].append(ba["bead_id"])

    # ── Derive bead connectivity from the underlying AA bonds ────────────────
    # Two beads are bonded iff any atom in one is chemically bonded (in the AA
    # topology) to any atom in the other. This is correct for every mapping
    # style — linear backbone, monomer-split (A-B-A-B), or pendant side beads —
    # because it follows the real chemistry instead of assuming which bead type
    # is "the backbone".
    atom_to_bead = {}
    for bid, aset in bead_atoms_of.items():
        for a in aset:
            atom_to_bead[a] = bid

    bead_adj = defaultdict(set)
    for (a, b) in bonds:                      # AA bonds (global atom ids)
        ba, bb = atom_to_bead.get(a), atom_to_bead.get(b)
        if ba is not None and bb is not None and ba != bb:
            bead_adj[ba].add(bb)
            bead_adj[bb].add(ba)

    cg_bonds  = []
    cg_angles = []

    # Bonds: each unique bead pair once
    seen_bonds = set()
    for bi in sorted(bead_adj):
        for bj in bead_adj[bi]:
            key = (min(bi, bj), max(bi, bj))
            if key not in seen_bonds:
                seen_bonds.add(key)
                cg_bonds.append(key)

    # Angles: every path bi-bj-bk through a central bead bj (bi < bk to dedupe)
    for bj in sorted(bead_adj):
        nbrs = sorted(bead_adj[bj])
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                cg_angles.append((nbrs[x], bj, nbrs[y]))

    return cg_atoms, cg_bonds, cg_angles, box

# ── Write CG data ─────────────────────────────────────────────────────────────

def write_cg_data(cg_atoms, cg_bonds, cg_angles, box, bead_map_def, polymer_name, output_path):
    bead_types   = bead_map_def["bead_types"]
    n_bead_types = len(bead_types)
    backbone_bt_id = next(
        (bt["bead_type_id"] for bt in bead_types if bt.get("is_backbone", False)),
        bead_types[0]["bead_type_id"]
    )

    bt_mass = {}
    for bt in bead_types:
        tid = bt["bead_type_id"]
        ms  = [a["mass"] for a in cg_atoms if a["bead_type"] == tid]
        bt_mass[tid] = float(np.mean(ms)) if ms else 12.011

    bead_type_lookup = {a["bead_id"]: a["bead_type"] for a in cg_atoms}

    # Bond types assigned by (sorted) bead-type pair — works for backbone,
    # monomer-split and pendant bonds alike.
    def _bond_key(b1, b2):
        t1 = bead_type_lookup.get(b1, backbone_bt_id)
        t2 = bead_type_lookup.get(b2, backbone_bt_id)
        return tuple(sorted((t1, t2)))

    bond_type_map = {}
    for (b1, b2) in cg_bonds:
        k = _bond_key(b1, b2)
        if k not in bond_type_map:
            bond_type_map[k] = len(bond_type_map) + 1
    n_bond_types = max(1, len(bond_type_map))

    # Angle types assigned by (normalised) bead-type triplet (central fixed,
    # ends symmetric). Computed up-front so the header count is correct.
    def _angle_key(b1, b2, b3):
        t1 = bead_type_lookup.get(b1, backbone_bt_id)
        t2 = bead_type_lookup.get(b2, backbone_bt_id)
        t3 = bead_type_lookup.get(b3, backbone_bt_id)
        ends = tuple(sorted((t1, t3)))
        return (ends[0], t2, ends[1])

    angle_type_map = {}
    for (b1, b2, b3) in cg_angles:
        k = _angle_key(b1, b2, b3)
        if k not in angle_type_map:
            angle_type_map[k] = len(angle_type_map) + 1
    n_angle_types = max(1, len(angle_type_map))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"LAMMPS CG data file -- {polymer_name} bead-mapped\n\n")
        f.write(f"   {len(cg_atoms)} atoms\n")
        f.write(f"   {len(cg_bonds)} bonds\n")
        f.write(f"   {len(cg_angles)} angles\n\n")
        f.write(f"   {n_bead_types} atom types\n")
        f.write(f"   {n_bond_types} bond types\n")
        f.write(f"   {n_angle_types} angle types\n\n")
        f.write(f"   {box.xlo:.6f}   {box.xhi:.6f} xlo xhi\n")
        f.write(f"   {box.ylo:.6f}   {box.yhi:.6f} ylo yhi\n")
        f.write(f"   {box.zlo:.6f}   {box.zhi:.6f} zlo zhi\n")

        f.write("\nMasses\n\n")
        for bt in bead_types:
            f.write(f"   {bt['bead_type_id']}  {bt_mass[bt['bead_type_id']]:.6f}"
                    f"  # {bt['name']} - {bt['description']}\n")

        f.write("\nAtoms # molecular\n\n")
        for a in sorted(cg_atoms, key=lambda x: x["bead_id"]):
            f.write(f"   {a['bead_id']:6d}  {a['mol']:5d}  {a['bead_type']:3d}"
                    f"  {a['x']:14.6f}  {a['y']:14.6f}  {a['z']:14.6f}\n")

        # Bond types assigned by (sorted) bead-type pair.
        f.write("\nBonds\n\n")
        for bid, (b1, b2) in enumerate(cg_bonds, 1):
            bond_type = bond_type_map[_bond_key(b1, b2)]
            f.write(f"   {bid:6d}   {bond_type}   {b1:6d}   {b2:6d}\n")

        f.write("\nAngles\n\n")
        for aid, (b1, b2, b3) in enumerate(cg_angles, 1):
            atype = angle_type_map[_angle_key(b1, b2, b3)]
            f.write(f"   {aid:6d}   {atype}   {b1:6d}   {b2:6d}   {b3:6d}\n")

    return output_path


# ── Public entry point ────────────────────────────────────────────────────────

def build_cg_data(aa_data_path, output_path, poly_def, bead_map_def, n_chains, n_monomers):
    atoms, bonds, box, masses_map = read_aa_data(aa_data_path)

    cg_atoms, cg_bonds, cg_angles, box = apply_bead_mapping(
        atoms, bonds, box, masses_map,
        poly_def, bead_map_def, n_chains, n_monomers,
    )

    write_cg_data(cg_atoms, cg_bonds, cg_angles, box,
                  bead_map_def, poly_def.get("polymer_name", "polymer"), output_path)

    n_beads_per_chain = len(cg_atoms) // max(n_chains, 1)

    print(f"\n✅  CG data written: {output_path}")
    print(f"   AA atoms  : {len(atoms)}")
    print(f"   CG beads  : {len(cg_atoms)}  ({n_beads_per_chain}/chain)")
    print(f"   CG bonds  : {len(cg_bonds)}")
    print(f"   CG angles : {len(cg_angles)}")
    print(f"   Bead types: {len(bead_map_def['bead_types'])}")

    return {
        "cg_data_path":    output_path,
        "n_beads":         len(cg_atoms),
        "n_bonds":         len(cg_bonds),
        "n_angles":        len(cg_angles),
        "n_bead_types":    len(bead_map_def["bead_types"]),
        "beads_per_chain": n_beads_per_chain,
    }
