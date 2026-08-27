"""
cg_constructor.py
=================
Coarse-grained LAMMPS data file and input script builder for CGMas.

Two public functions:
  build_cg_initial_data(...)  – generate CG .data file from bead mapping + potential results
  build_cg_in_file(...)       – generate CG LAMMPS input script from in.CGEQ template
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

AVOGADRO    = 6.02214076e23
CM3_TO_A3   = 1.0e24

DEFAULT_T_INIT_CG   = 300
DEFAULT_T_MAX_CG    = 500
DEFAULT_N_ANNEAL_CG = 1
DEFAULT_EQUIL_PS_CG = 1000
DEFAULT_ANNEAL_SPAN_CG = 200   # K  Tfin - Tini fallback when only Tini is given
STEPS_PER_PS_CG     = 1000


# ─────────────────────────────────────────────────────────────────────────────
# MASS & BOX HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bead_type_masses(poly_def: dict, bead_def: dict, opls_atoms: dict) -> dict:
    """
    Compute mass for each bead type by summing atomic masses of member atoms.
    Supports both homopolymer (old format) and copolymer (monomer_types format).
    """
    from polymer import normalize_poly_def
    poly_def_n    = normalize_poly_def(poly_def, 1)   # n_monomers=1 just to get structure
    monomer_types = poly_def_n["monomer_types"]
    has_mono_type = any("monomer_type" in bt for bt in bead_def.get("bead_types", []))

    bead_masses = {}
    for bt in bead_def.get("bead_types", []):
        bt_id = bt["bead_type_id"]
        lids  = bt.get("atom_local_ids") or bt.get("local_ids_in_monomer", [])
        mpb   = bt.get("monomers_per_bead") or bt.get("repeating_monomers", 1) or 1

        # Determine which monomer_type this bead belongs to
        lbl = bt.get("monomer_type", list(monomer_types.keys())[0])
        mdef = monomer_types.get(lbl, list(monomer_types.values())[0])
        lid_to_mass = {a["local_id"]: opls_atoms.get(a["opls_id"], {}).get("mass", 12.011)
                       for a in mdef.get("monomer_atoms", [])}

        mass = sum(lid_to_mass.get(lid, 12.011) for lid in lids)
        if mass <= 0 and lid_to_mass:
            mass = sum(lid_to_mass.values())
        bead_masses[bt_id] = mass * mpb
    return bead_masses


def _calculate_cg_box_size(n_chains: int, n_monomers: int,
                            bead_def: dict, bead_masses: dict,
                            density: float) -> float:
    """Return CG box side length in Å from target density."""
    if not density or density <= 0:
        density = 0.9
    total_bead_mass = sum(bead_masses.values()) if bead_masses else 12.011
    if total_bead_mass <= 0:
        total_bead_mass = 12.011
    mass_per_chain = total_bead_mass * n_monomers
    total_mass = n_chains * mass_per_chain
    volume_cm3 = total_mass / (density * AVOGADRO)
    return (volume_cm3 * CM3_TO_A3) ** (1.0 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# POSITION GENERATOR  –  simple random-walk chain placement
# ─────────────────────────────────────────────────────────────────────────────

def _place_cg_chains(n_chains: int, beads_per_chain: int,
                     bond_r0: float, box_L: float,
                     seed: int | None = None) -> list[np.ndarray]:
    """
    Generate bead positions for all chains as a random walk.
    Returns flat list of (x, y, z) arrays in chain order.
    """
    rng = np.random.default_rng(seed)
    all_positions: list[np.ndarray] = []

    grid_n    = int(np.ceil(n_chains ** (1.0 / 3.0)))
    cell_size = box_L / grid_n
    grid_pts  = [(i, j, k)
                 for i in range(grid_n)
                 for j in range(grid_n)
                 for k in range(grid_n)]
    rng.shuffle(grid_pts)

    for ci in range(n_chains):
        gi, gj, gk = grid_pts[ci % len(grid_pts)]
        origin = (np.array([gi, gj, gk], dtype=float) + 0.5) * cell_size
        origin += rng.uniform(-0.15 * cell_size, 0.15 * cell_size, size=3)

        pos = origin.copy()
        chain_pos: list[np.ndarray] = [pos.copy()]

        for _ in range(beads_per_chain - 1):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            pos = pos + bond_r0 * direction
            # wrap into box
            pos = pos % box_L
            chain_pos.append(pos.copy())

        all_positions.extend(chain_pos)

    return all_positions


# ─────────────────────────────────────────────────────────────────────────────
# DATA FILE WRITER
# ─────────────────────────────────────────────────────────────────────────────

def build_cg_initial_data(
    poly_def:          dict,
    bead_def:          dict,
    potential_results: dict,
    opls_atoms:        dict,
    n_chains:          int   = 20,
    n_monomers:        int   = 10,
    density:           float = 0.9,
    output_path:       str   = "cg_initial.data",
    seed:              int | None = None,
    cg_topology:       dict  = None,
) -> dict:
    """
    Build a CG LAMMPS data file (atom_style molecular / angle).

    Parameters
    ----------
    poly_def          : original AA polymer topology dict
    bead_def          : normalized bead mapping dict (keys: atom_local_ids, monomers_per_bead)
    potential_results : output of compute_cg_potentials  {bond, angle, lj}
    opls_atoms        : OPLS-AA atom table from load_opls
    n_chains          : number of CG chains
    n_monomers        : monomers per chain (same as AA)
    density           : target density g/cm³
    output_path       : where to write the .data file
    seed              : random seed

    Returns
    -------
    dict with metadata
    """
    bead_types = bead_def["bead_types"]
    n_bead_types = len(bead_types)
    max_mpb      = max(bt.get("monomers_per_bead", 1) for bt in bead_types)
    n_groups     = n_monomers // max_mpb   # monomer groups per chain

    # ── Masses ───────────────────────────────────────────────────────────────
    bead_masses = _bead_type_masses(poly_def, bead_def, opls_atoms)

    # ── Box size ──────────────────────────────────────────────────────────────
    box_L = _calculate_cg_box_size(n_chains, n_monomers, bead_def, bead_masses, density)

    # ── Beads per chain ───────────────────────────────────────────────────────
    # Derive CG bead sequence for one chain (respects copolymer sequence)
    from polymer import normalize_poly_def, derive_cg_bead_sequence
    poly_def_n      = normalize_poly_def(poly_def, n_monomers)
    aa_sequence     = poly_def_n["sequence"]
    cg_bead_seq_one = derive_cg_bead_sequence(aa_sequence, bead_def)
    beads_per_chain = len(cg_bead_seq_one)
    total_beads     = n_chains * beads_per_chain

    # ── Positions ─────────────────────────────────────────────────────────────
    # Get bond r0: multi-type uses dict of dicts, single-type uses flat dict
    _bond_res = potential_results["bond"]
    if _bond_res and isinstance(next(iter(_bond_res)), tuple):
        # multi-type: use backbone-backbone (1,1) or first entry
        _first_key = (1,1) if (1,1) in _bond_res else next(iter(_bond_res))
        bond_r0 = _bond_res[_first_key]["r0"]
    else:
        bond_r0 = _bond_res.get("r0", 4.0)
    if not bond_r0 or bond_r0 <= 0:
        bond_r0 = 4.0  # fallback
    positions  = _place_cg_chains(n_chains, beads_per_chain, bond_r0, box_L, seed)

    # ── Build atoms list: (bead_id, mol_id, bead_type_id, x, y, z) ───────────
    atoms: list[tuple] = []
    bead_id = 1
    for ci in range(n_chains):
        mol_id = ci + 1
        for bi, bt_id in enumerate(cg_bead_seq_one):
            pos = positions[ci * beads_per_chain + bi]
            atoms.append((bead_id, mol_id, bt_id, pos[0], pos[1], pos[2]))
            bead_id += 1

    # ── Map bead_type_id -> sequential type index ────────────────────────────
    bead_type_id_map = {bt["bead_type_id"]: i+1 for i, bt in enumerate(bead_types)}

    # ── Determine which bead types lie on the backbone ──────────────────────
    # A bead is on the backbone iff it contains its monomer's head_atom or
    # tail_atom (the inter-monomer connection points). This is derived from the
    # actual chemistry rather than trusting an LLM is_backbone flag, and it is
    # correct for every case:
    #   - homopolymer pendant maps (PMP): the backbone bead holds head/tail, the
    #     side-chain bead does not  -> one backbone type -> comb;
    #   - 1-bead-per-monomer linear (co)polymers (PS, SBS): every bead holds its
    #     monomer's head & tail     -> all bead types are backbone -> linear chain.
    # An explicit is_backbone flag, when present, is honoured as an override.
    from polymer import normalize_poly_def as _norm_pd
    _pdn = _norm_pd(poly_def, n_monomers)
    _mtypes = _pdn.get("monomer_types", {})

    def _is_backbone_bead(bt):
        if "is_backbone" in bt:
            return bool(bt["is_backbone"])
        lbl  = bt.get("monomer_type")
        mdef = _mtypes.get(lbl) if lbl is not None else next(iter(_mtypes.values()), {})
        if not mdef:
            return True
        ht  = {mdef.get("head_atom"), mdef.get("tail_atom")}
        ids  = set(bt.get("atom_local_ids") or bt.get("local_ids_in_monomer", []))
        return bool(ht & ids)

    backbone_bt_ids = {bt["bead_type_id"] for bt in bead_types if _is_backbone_bead(bt)}
    if not backbone_bt_ids:                      # degenerate fallback
        backbone_bt_ids = {bead_types[0]["bead_type_id"]}
    backbone_bt_id = next(
        (bt["bead_type_id"] for bt in bead_types if bt["bead_type_id"] in backbone_bt_ids),
        bead_types[0]["bead_type_id"]
    )
    pendant_bt_ids = [bt["bead_type_id"] for bt in bead_types
                      if bt["bead_type_id"] not in backbone_bt_ids]

    # Build bond_pair_to_type from cg_topology or default
    bond_pair_to_type: dict[tuple, int] = {}
    if cg_topology and cg_topology.get("bond_type_pairs"):
        for i, pair in enumerate(cg_topology["bond_type_pairs"], 1):
            key = tuple(sorted(pair))
            bond_pair_to_type[key] = i
    else:
        bond_pair_to_type[tuple(sorted([backbone_bt_id, backbone_bt_id]))] = 1
        for j, pbt in enumerate(pendant_bt_ids, 2):
            bond_pair_to_type[tuple(sorted([backbone_bt_id, pbt]))] = j

    # Build angle_triplet_to_type from cg_topology or default
    # TWO-PASS: forward keys first, then reverse only if not already a forward key.
    # Single-pass with reversed() causes later explicit types to overwrite earlier
    # reverse entries (e.g. [2,1,1]=3 stored as rev of [1,1,2] overwrites [1,1,2]=2).
    angle_triplet_to_type: dict[tuple, int] = {}
    if cg_topology and cg_topology.get("angle_type_triplets"):
        for i, triplet in enumerate(cg_topology["angle_type_triplets"], 1):
            angle_triplet_to_type[tuple(triplet)] = i          # pass 1: forward
        for i, triplet in enumerate(cg_topology["angle_type_triplets"], 1):
            rev = tuple(reversed(triplet))
            if rev not in angle_triplet_to_type:               # pass 2: reverse only if new
                angle_triplet_to_type[rev] = i
    else:
        angle_triplet_to_type[(backbone_bt_id, backbone_bt_id, backbone_bt_id)] = 1

    # Map (chain, seq_position) -> bead_id
    bead_lookup_flat: dict[tuple, int] = {}
    bid = 1
    for ci in range(n_chains):
        for bi in range(beads_per_chain):
            bead_lookup_flat[(ci, bi)] = bid
            bid += 1

    # Also build (chain, group_index, bt_id) lookup using cg_bead_seq
    bead_lookup: dict[tuple, int] = {}
    bid = 1
    for ci in range(n_chains):
        for bi, bt_id in enumerate(cg_bead_seq_one):
            bead_lookup[(ci, bi, bt_id)] = bid
            # Compute group index (how many backbone beads before this)
            bid += 1

    bonds: list[tuple] = []   # (bond_type, bead_i, bead_j)
    angles: list[tuple] = []  # (angle_type, b1, b2, b3)

    def _bond_type(bt_i: int, bt_j: int) -> int:
        key = tuple(sorted([bt_i, bt_j]))
        return bond_pair_to_type.get(key, 1)

    def _angle_type(bt_i: int, bt_j: int, bt_k: int) -> int:
        # An angle is symmetric under end-swap: (i,j,k) and (k,j,i) are the same
        # physical angle. The LLM-supplied topology sometimes lists both
        # orientations as separate triplets (e.g. [1,1,2] AND [2,1,1]), which
        # would otherwise split one angle across two type ids — leaving one with
        # no potential coeff. Collapse both orientations to the smallest id.
        cands = [angle_triplet_to_type.get((bt_i, bt_j, bt_k)),
                 angle_triplet_to_type.get((bt_k, bt_j, bt_i))]
        cands = [c for c in cands if c is not None]
        return min(cands) if cands else 1

    # ── Backbone-aware (comb) connectivity ──────────────────────────────────
    # The CG bead sequence lists beads monomer-group by monomer-group; within a
    # group the backbone bead precedes its pendant bead(s). Bonding *consecutive*
    # sequence entries (the previous behaviour) threaded pendant beads INTO the
    # main chain, producing a linear A-B-A-B copolymer instead of an A backbone
    # with B pendants. That corrupted the bond/angle topology (all collapsed to
    # one type) and the density. Instead we build the same comb topology the
    # bead mapper produces: chain the backbone beads together and hang each
    # pendant off the most recent backbone bead in its group. When every bead is
    # a backbone bead (single-bead-per-monomer linear chains) this reduces to the
    # old consecutive bonding, so simple polymers are unaffected.
    is_bb_pos = [bt_id in backbone_bt_ids for bt_id in cg_bead_seq_one]

    for ci in range(n_chains):
        chain_bids  = [ci * beads_per_chain + bi + 1 for bi in range(beads_per_chain)]
        chain_btids = cg_bead_seq_one  # same for every chain

        # Per-chain adjacency between sequence positions
        adj: dict[int, set] = {bi: set() for bi in range(beads_per_chain)}
        last_bb = None
        for bi in range(beads_per_chain):
            if is_bb_pos[bi]:
                if last_bb is not None:                 # backbone–backbone bond
                    adj[last_bb].add(bi); adj[bi].add(last_bb)
                last_bb = bi
            else:                                       # pendant: attach to its backbone bead
                anchor = last_bb
                if anchor is None:                      # pendant before any backbone (rare)
                    anchor = next((j for j in range(bi + 1, beads_per_chain)
                                   if is_bb_pos[j]), None)
                if anchor is not None:
                    adj[anchor].add(bi); adj[bi].add(anchor)

        # Bonds: each unique adjacent pair once
        seen = set()
        for bi in sorted(adj):
            for bj in sorted(adj[bi]):
                key = (min(bi, bj), max(bi, bj))
                if key in seen:
                    continue
                seen.add(key)
                btype = _bond_type(chain_btids[bi], chain_btids[bj])
                bonds.append((btype, chain_bids[bi], chain_bids[bj]))

        # Angles: every path end–centre–end through a central bead
        for bj in sorted(adj):
            nbrs = sorted(adj[bj])
            for x in range(len(nbrs)):
                for y in range(x + 1, len(nbrs)):
                    bi_, bk_ = nbrs[x], nbrs[y]
                    atype = _angle_type(chain_btids[bi_], chain_btids[bj], chain_btids[bk_])
                    angles.append((atype, chain_bids[bi_], chain_bids[bj], chain_bids[bk_]))

    # Remap bond/angle type IDs to sequential (1,2,3,...) based on actually used types
    used_bond_types  = sorted(set(b[0] for b in bonds))  if bonds  else [1]
    used_angle_types = sorted(set(a[0] for a in angles)) if angles else [1]
    n_bond_types     = len(used_bond_types)
    n_angle_types    = len(used_angle_types)
    btype_remap      = {old_t: new_t for new_t, old_t in enumerate(used_bond_types, 1)}
    atype_remap      = {old_t: new_t for new_t, old_t in enumerate(used_angle_types, 1)}
    bonds            = [(btype_remap.get(b[0], 1), b[1], b[2]) for b in bonds]
    angles           = [(atype_remap.get(a[0], 1), a[1], a[2], a[3]) for a in angles]

    # ── Write data file ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        poly_name = poly_def.get("polymer_name", "polymer")
        f.write(f"# LAMMPS CG data file -- {poly_name} ({n_chains} chains x {n_monomers} monomers)\n\n")

        f.write(f"{total_beads} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n\n")

        f.write(f"{n_bead_types} atom types\n")
        f.write(f"{n_bond_types} bond types\n")
        f.write(f"{n_angle_types} angle types\n\n")

        f.write(f"0.000   {box_L:.4f} xlo xhi\n")
        f.write(f"0.000   {box_L:.4f} ylo yhi\n")
        f.write(f"0.000   {box_L:.4f} zlo zhi\n")

        f.write("\nMasses\n\n")
        for bt in bead_types:
            bt_id   = bt["bead_type_id"]
            bt_name = bt.get("name", f"bead{bt_id}")
            f.write(f"{bt_id}  {bead_masses[bt_id]:.4f}  # {bt_name}\n")

        f.write("\nAtoms # molecular\n\n")
        for (bid, mol_id, bt_id, x, y, z) in atoms:
            f.write(f"  {bid:8d}  {mol_id:6d}  {bt_id:3d}  {x:14.6f}  {y:14.6f}  {z:14.6f}\n")

        f.write("\nBonds\n\n")
        for i, (btype, b1, b2) in enumerate(bonds, 1):
            f.write(f"  {i:8d}  {btype}  {b1:8d}  {b2:8d}\n")

        f.write("\nAngles\n\n")
        for i, (atype, b1, b2, b3) in enumerate(angles, 1):
            f.write(f"  {i:8d}  {atype}  {b1:8d}  {b2:8d}  {b3:8d}\n")

    print(f"\n✅  CG data file written: {output_path}")
    print(f"   Polymer    : {poly_name}")
    print(f"   Chains     : {n_chains} x {n_monomers} monomers  ({beads_per_chain} beads/chain)")
    print(f"   Total beads: {total_beads}  |  Bonds: {len(bonds)}  |  Angles: {len(angles)}")
    print(f"   Bead types : {n_bead_types}  |  Bond types: {n_bond_types}  |  Angle types: {n_angle_types}")
    print(f"   Box size   : {box_L:.4f} Å  (density {density} g/cm³)")

    return {
        "output_path":       output_path,
        "box_L":             round(box_L, 4),
        "total_beads":       total_beads,
        "beads_per_chain":   beads_per_chain,
        "n_bead_types":      n_bead_types,
        "n_bond_types":      n_bond_types,
        "n_angle_types":     n_angle_types,
        "bead_masses":       bead_masses,
        "n_chains":          n_chains,
        "n_monomers":        n_monomers,
        "density":           density,
        "used_bond_types":   used_bond_types,
        "used_angle_types":  used_angle_types,
        "bond_type_remap":   btype_remap,
        "angle_type_remap":  atype_remap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IN-FILE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_cg_in_file(
    template_path:     str,
    data_filename:     str,
    potential_results: dict,
    bead_def:          dict,
    output_path:       str,
    t_init:            int   = DEFAULT_T_INIT_CG,
    t_max:             int   = DEFAULT_T_MAX_CG,
    n_anneal:          int   = DEFAULT_N_ANNEAL_CG,
    equil_ps:          int   = DEFAULT_EQUIL_PS_CG,
    used_bond_types:   list  = None,
    used_angle_types:  list  = None,
    bond_type_remap:   dict  = None,
    angle_type_remap:  dict  = None,
) -> str:
    """
    Generate CG LAMMPS input script from in.CGEQ template.
    Supports both single-type and multi-type potential_results.
    """
    # Normalise the requested temperatures first: values coming from a user
    # query may be str/float, and Tfin must stay above Tini for the ramp.
    try:
        from simulator import normalize_temps as _norm_T
        t_init, t_max = _norm_T(t_init, t_max, label="CG")
    except Exception:
        t_init, t_max = int(round(float(t_init))), int(round(float(t_max)))
        if t_max < t_init:
            t_max = t_init + DEFAULT_ANNEAL_SPAN_CG

    bond_results  = potential_results["bond"]
    angle_results = potential_results["angle"]
    lj_results    = potential_results["lj"]

    # Normalize keys: tuples may be stored as lists or strings after JSON serialization
    def _to_tuple(k):
        if isinstance(k, (list, tuple)):
            return tuple(int(x) for x in k)
        if isinstance(k, str):
            # "(1, 2)" -> (1, 2)
            import ast
            return tuple(int(x) for x in ast.literal_eval(k))
        return k

    def _is_multi_type(d: dict) -> bool:
        """Return True if the dict uses tuple/list/tuple-string keys (multi bead-type result).
        Handles raw tuples, lists, "(1, 2)" strings, and "[1, 2]" strings
        (the last form is produced by _serialize_potential_results for MemorySaver compat)."""
        if not d:
            return False
        k = next(iter(d))
        if isinstance(k, (tuple, list)):
            return True
        if isinstance(k, str) and (k.startswith("(") or k.startswith("[")):
            return True
        return False

    # Detect multi-type BEFORE normalizing to avoid crashing on single-type string keys
    is_multi = _is_multi_type(bond_results)

    # Only normalize keys when it is actually a multi-type result
    if is_multi:
        bond_results  = {_to_tuple(k): v for k, v in bond_results.items()}
        angle_results = {_to_tuple(k): v for k, v in angle_results.items()}
        lj_results    = {_to_tuple(k): v for k, v in lj_results.items()}

    equil_steps  = equil_ps * STEPS_PER_PS_CG
    bead_types   = bead_def["bead_types"]
    n_bead_types = len(bead_types)

    # Build ordered lists from stored order or dict keys
    if is_multi:
        raw_border  = potential_results.get("bond_type_order",  list(bond_results.keys()))
        raw_aorder  = potential_results.get("angle_type_order", list(angle_results.keys()))
        raw_ljorder = potential_results.get("lj_type_order",    list(lj_results.keys()))
        bond_order  = [_to_tuple(k) for k in raw_border]
        angle_order = [_to_tuple(k) for k in raw_aorder]
        lj_order    = [_to_tuple(k) for k in raw_ljorder]

        # Filter to only types actually used in data file (avoid out-of-bounds)
        if used_bond_types:
            # used_bond_types are original topology IDs; remap to sequential
            remap_b = bond_type_remap or {}
            valid_bond_ids = set(remap_b.get(t, t) for t in used_bond_types)
            bond_order = [bt for bt in bond_order if bt in bond_results]
            # Reindex bond_order to match data file sequential IDs
            bond_order_reindexed = []
            for orig_bt in bond_order:
                bond_order_reindexed.append(orig_bt)
            bond_order = bond_order_reindexed

        if used_angle_types:
            remap_a = angle_type_remap or {}
            angle_order = [at for at in angle_order if at in angle_results]
    else:
        bond_order  = [None]
        angle_order = [None]
        lj_order    = [(i+1, j+1) for i in range(n_bead_types) for j in range(i, n_bead_types)]

    with open(template_path) as f:
        lines = f.readlines()

    patched: list[str] = []
    seen_tini = seen_tfin = False

    for line in lines:
        # read_data
        if re.match(r'\s*read_data', line):
            patched.append(f"read_data       {data_filename}\n")
            continue

        # bond_coeff – match any bond_coeff line (not just 1) to avoid leftovers
        if re.match(r'\s*bond_coeff\s+', line):
            # Only emit once when we hit the first bond_coeff line
            if not any('bond_coeff' in p for p in patched):
                if is_multi:
                    remap_b = bond_type_remap or {}
                    for topo_id, btype in enumerate(bond_order, 1):
                        if btype not in bond_results:
                            continue
                        if used_bond_types and topo_id not in used_bond_types:
                            continue
                        seq_id = remap_b.get(topo_id, topo_id)
                        p = bond_results[btype]
                        patched.append(
                            f"bond_coeff      {seq_id} {p['k']:.6f} {p['r0']:.6f}"
                            f"  # {list(btype)}\n")
                else:
                    p = bond_results
                    if isinstance(p, dict) and "k" in p and "r0" in p:
                        patched.append(f"bond_coeff      1 {p['k']:.6f} {p['r0']:.6f}\n")
            continue  # always skip original line

        # angle_coeff
        if re.match(r'\s*angle_coeff\s+', line):
            if not any('angle_coeff' in p for p in patched):
                if is_multi:
                    # Build topology_id → seq_id mapping from remap
                    # topology_id = 1-based position in angle_order list
                    remap_a = angle_type_remap or {}
                    for topo_id, atype in enumerate(angle_order, 1):
                        if atype not in angle_results:
                            continue
                        # Skip if this topology type was not used in the data file
                        if used_angle_types and topo_id not in used_angle_types:
                            continue
                        seq_id = remap_a.get(topo_id, topo_id)
                        p = angle_results[atype]
                        patched.append(
                            f"angle_coeff     {seq_id} {p['k']:.6f} {p['theta0']:.6f}"
                            f"  # {list(atype)}\n")
                else:
                    p = angle_results
                    if isinstance(p, dict) and "k" in p and "theta0" in p:
                        patched.append(f"angle_coeff     1 {p['k']:.6f} {p['theta0']:.6f}\n")
            continue

        # pair_coeff
        if re.match(r'\s*pair_coeff\s+', line):
            if not any('pair_coeff' in p for p in patched):
                if is_multi:
                    for lj_key in lj_order:
                        p = lj_results[lj_key]
                        ti, tj = lj_key
                        patched.append(f"pair_coeff      {ti} {tj} {p['epsilon']:.6f} {p['sigma']:.6f}  # {list(lj_key)}\n")
                else:
                    p = lj_results
                    for i in range(1, n_bead_types + 1):
                        for j in range(i, n_bead_types + 1):
                            patched.append(f"pair_coeff      {i} {j} {p['epsilon']:.6f} {p['sigma']:.6f}\n")
            continue

        # variables
        if re.match(r'\s*variable\s+Tini\s+equal', line, re.IGNORECASE):
            patched.append(f"variable Tini   equal {t_init}\n")
            seen_tini = True; continue
        if re.match(r'\s*variable\s+Tfin\s+equal', line, re.IGNORECASE):
            patched.append(f"variable Tfin   equal {t_max}\n")
            seen_tfin = True; continue
        if re.match(r'\s*variable\s+equil\s+equal', line):
            patched.append(f"variable equil  equal {equil_steps}\n"); continue

        patched.append(line)

    # If the CGEQ template defines no Tini/Tfin variable, inject them so the
    # requested temperature is never silently dropped.
    if not (seen_tini and seen_tfin):
        try:
            from simulator import _ensure_temp_vars as _inj
            patched = _inj(patched, t_init, t_max, seen_tini, seen_tfin)
        except Exception:
            inj = ([] if seen_tini else [f"variable Tini   equal {t_init}\n"]) + \
                  ([] if seen_tfin else [f"variable Tfin   equal {t_max}\n"])
            patched = inj + patched

    template_text = "".join(patched)

    print(f"[CG Constructor] in.CGEQ  ->  Tini = {t_init} K | Tfin = {t_max} K "
          f"| {n_anneal} annealing cycle(s) | {equil_ps} ps EQ")

    # Multi-anneal replication
    if n_anneal > 1:
        anneal_block_tpl = """# ----- Temp increase  [cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tini}} ${{Tfin}} 1000.0
dump           1 all atom 10000 cg_heating_{cyc}.lammpstrj
run            100000

unfix     1
undump    1

# ----- NVT  [holding, cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tfin}} ${{Tfin}} 1000.0
dump           1 all atom 10000 cg_holding_{cyc}.lammpstrj
run            100000

unfix     1
undump    1

# ----- Temp decrease  [cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tfin}} ${{Tini}} 1000.0
dump           1 all atom 10000 cg_cooling_{cyc}.lammpstrj
run            100000

unfix     1
undump    1

# ----- NVT  [relaxing, cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tini}} ${{Tini}} 1000.0
dump           1 all atom 10000 cg_relaxing_{cyc}.lammpstrj
run            100000

unfix     1
undump    1
"""
        m = re.search(r'(# ----- Temp increase.*?)(# ----- NPT)', template_text, re.DOTALL)
        if m:
            anneal_blocks = "".join(anneal_block_tpl.format(cyc=cyc) for cyc in range(1, n_anneal+1))
            template_text = template_text[:m.start(1)] + anneal_blocks + template_text[m.start(2):]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template_text)

    print(f"\n✅  CG in-file written: {output_path}")
    return output_path
