"""
aa_constructor.py
=================
OPLS-AA Amorphous Polymer LAMMPS Data File Generator

Receives a polymer topology definition (poly_def dict) from the LLM agent,
looks up OPLS-AA parameters from the force-field text files, generates
3-D coordinates via **bond-graph BFS traversal with ring detection**, and writes
a LAMMPS data file in 'full' atom style.

Coordinate generation strategy
───────────────────────────────
Coordinates are built atom-by-atom by traversing the intra-monomer bond graph in
BFS order starting from the inter-monomer attachment point (head_atom).  For each
newly visited atom the position is determined by:

  • Z-matrix placement  – bond length and valence angle looked up from the OPLS-AA
    parameter files; torsion angle drawn randomly (amorphous bulk).
  • Ring placement      – when a ring is first entered, all ring atoms are laid out
    as a regular polygon (correct n-membered ring geometry) oriented by a random
    rotation around the attachment bond.  This handles benzene, imidazole, pyrrole,
    etc. at arbitrary side-chain depth.

Because every step uses OPLS-derived bond lengths and angles the method is
automatically correct for heterogeneous backbones (C-O, C-N, …) and arbitrarily
deep side chains.

Topology JSON schema expected from the LLM
──────────────────────────────────────────
{
  "polymer_name":  str,
  "monomer_atoms": [
    { "local_id": int,   # 1-based ID within one repeat unit
      "opls_id":  str,   # e.g. "opls_145B"
      "charge":   float, # optional – overrides OPLS default
      "comment":  str  } # optional
  ],
  "intra_bonds":  [[i, j], ...],  # bonds within one repeat unit (local_ids)
  "head_atom":    int,            # local_id bonded to tail of *previous* monomer
  "tail_atom":    int,            # local_id bonded to head  of *next*     monomer
  "notes":        str             # optional force-field caveats
}
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict, deque
from itertools import combinations
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION DEFAULTS  (imported by cgmas.ipynb — do not duplicate there)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_CHAINS   = 20     # number of polymer chains per system
DEFAULT_N_MONOMERS = 10     # repeat units per chain
DEFAULT_DENSITY    = 0.9    # g/cm³  (amorphous polymer bulk target density)
DEFAULT_SEED       = 42     # random seed for reproducible coordinates
MAX_REVISIONS      = 2      # LLM reviewer retry limit

# ─────────────────────────────────────────────────────────────────────────────
# OPLS-AA PARAMETER PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_atom_params(path: str) -> dict:
    """
    Parse OPLS-AA atom.txt — supports both old and new format.

    New format (no opls_xxx prefix, bond_type can repeat):
      type  Z  mass  charge  ptype  sigma  epsilon  ;comment

    Old format (with unique opls_xxx prefix):
      opls_id  bond_type  Z  mass  charge  ptype  sigma  epsilon  ;comment

    Returns {unique_key: {opls_id, bond_type, mass, charge, sigma, epsilon, comment}}.

    For new-format duplicates (e.g. CT appears 8 times), unique keys are generated
    as  CT_1, CT_2, … so the LLM can pick the right variant by comment.
    For old-format entries the key is the opls_xxx string as before.
    """
    result = {}
    type_count: dict[str, int] = {}

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('[') or s.startswith(';'):
                continue
            p = s.split()
            if len(p) < 7:
                continue

            # Detect format: old format has opls_xxx as first token
            if p[0].lower().startswith('opls_'):
                # Old format: opls_id  bond_type  Z  mass  charge  ptype  sigma  epsilon
                if len(p) < 8:
                    continue
                try:
                    opls_id   = p[0]
                    bond_type = p[1]
                    mass      = float(p[3])
                    charge    = float(p[4])
                    sigma     = float(p[6])
                    epsilon   = float(p[7])
                except (ValueError, IndexError):
                    continue
                comment = ' '.join(p[8:]).lstrip(';').strip() if len(p) > 8 else ''
                result[opls_id] = {
                    'opls_id':   opls_id,
                    'bond_type': bond_type,
                    'mass':      mass,
                    'charge':    charge,
                    'sigma':     sigma,
                    'epsilon':   epsilon,
                    'comment':   comment,
                }

            else:
                # New format: bond_type  Z  mass  charge  ptype  sigma  epsilon  ;comment
                try:
                    bond_type = p[0]
                    mass      = float(p[2])
                    charge    = float(p[3])
                    # p[4] is ptype ('A' or 'D') — skip
                    sigma     = float(p[5])
                    epsilon   = float(p[6])
                except (ValueError, IndexError):
                    continue
                comment = ' '.join(p[7:]).lstrip(';').strip() if len(p) > 7 else ''

                # Generate unique key: CT_1, CT_2, …
                type_count[bond_type] = type_count.get(bond_type, 0) + 1
                unique_key = bond_type + '_' + str(type_count[bond_type])

                result[unique_key] = {
                    'opls_id':   unique_key,   # use generated key as opls_id
                    'bond_type': bond_type,
                    'mass':      mass,
                    'charge':    charge,
                    'sigma':     sigma,
                    'epsilon':   epsilon,
                    'comment':   comment,
                }

    return result


def parse_bond_params(path: str) -> dict:
    """Returns {(btype_i, btype_j): {r0 [Å], k [kcal/mol/Å²]}}. Symmetric."""
    result = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('[') or s.startswith(';'):
                continue
            p = s.split()
            if len(p) < 5:
                continue
            ti, tj = p[0], p[1]
            val = {'r0': float(p[3]), 'k': float(p[4])}
            result[(ti, tj)] = val
            result[(tj, ti)] = val
    return result


def parse_angle_params(path: str) -> dict:
    """Returns {(bi, bj, bk): {th0 [°], k [kcal/mol/rad²]}}. Symmetric."""
    result = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('[') or s.startswith(';'):
                continue
            p = s.split()
            if len(p) < 6:
                continue
            ti, tj, tk = p[0], p[1], p[2]
            val = {'th0': float(p[4]), 'k': float(p[5])}
            result[(ti, tj, tk)] = val
            result[(tk, tj, ti)] = val
    return result


def parse_dihedral_params(path: str) -> dict:
    """Returns {(bi,bj,bk,bl): [k1,k2,k3,k4]}. Symmetric (reversed key also stored)."""
    result = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('[') or s.startswith(';'):
                continue
            p = s.split()
            if len(p) < 9 or p[4] != 'opls':
                continue
            ti, tj, tk, tl = p[0], p[1], p[2], p[3]
            val = [float(p[5]), float(p[6]), float(p[7]), float(p[8])]
            result[(ti, tj, tk, tl)] = val
            result[(tl, tk, tj, ti)] = val
    return result


def load_opls(opls_dir: str) -> dict:
    return {
        'atom':     parse_atom_params(os.path.join(opls_dir, 'atom.txt')),
        'bond':     parse_bond_params(os.path.join(opls_dir, 'bond.txt')),
        'angle':    parse_angle_params(os.path.join(opls_dir, 'angle.txt')),
        'dihedral': parse_dihedral_params(os.path.join(opls_dir, 'dihedral.txt')),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK PARAMETERS (when not found in OPLS files)
# ─────────────────────────────────────────────────────────────────────────────

# Common bond defaults by element pair (using 1-letter element guess)
_BOND_DEFAULTS = {
    ('C', 'C'):  {'r0': 1.529, 'k': 268.0},
    ('C', 'H'):  {'r0': 1.090, 'k': 340.0},
    ('C', 'O'):  {'r0': 1.410, 'k': 320.0},
    ('C', 'N'):  {'r0': 1.449, 'k': 337.0},
    ('O', 'H'):  {'r0': 0.960, 'k': 553.0},
    ('N', 'H'):  {'r0': 1.010, 'k': 434.0},
}

_ANGLE_DEFAULTS = {
    ('C', 'C', 'C'):  {'th0': 112.7, 'k': 58.35},
    ('C', 'C', 'H'):  {'th0': 110.7, 'k': 37.50},
    ('H', 'C', 'H'):  {'th0': 107.8, 'k': 33.00},
    ('C', 'O', 'C'):  {'th0': 111.8, 'k': 60.00},
    ('C', 'C', 'O'):  {'th0': 109.5, 'k': 50.00},
    ('C', 'O', 'H'):  {'th0': 108.5, 'k': 55.00},
    ('C', 'N', 'H'):  {'th0': 113.9, 'k': 43.60},
    ('C', 'C', 'N'):  {'th0': 111.2, 'k': 58.35},
}

_DIHEDRAL_DEFAULTS = {
    ('C', 'C', 'C', 'C'):  [0.0, 1.740, -0.157, 0.279],
    ('H', 'C', 'C', 'H'):  [0.0, 0.0,    0.318,  0.0],
    ('H', 'C', 'C', 'C'):  [0.0, 0.0,    0.366,  0.0],
    ('C', 'C', 'O', 'C'):  [0.0, 0.0,    0.760,  0.0],
    ('H', 'C', 'O', 'C'):  [0.0, 0.0,    0.760,  0.0],
    ('C', 'O', 'C', 'H'):  [0.0, 0.0,    0.760,  0.0],
    ('C', 'C', 'N', 'H'):  [0.0, 0.0,    0.0,    0.0],
}


def _element_of(bond_type: str) -> str:
    """Guess single-letter element from bond_type name (CT→C, HC→H, OS→O, …)."""
    bt = bond_type.upper()
    if bt.startswith('H'):
        return 'H'
    if bt.startswith('O'):
        return 'O'
    if bt.startswith('N'):
        return 'N'
    if bt.startswith('S'):
        return 'S'
    if bt.startswith('F'):
        return 'F'
    if bt.startswith('P'):
        return 'P'
    if bt.startswith('C'):
        return 'C'
    return bt[0]


def lookup_bond(opls: dict, bt_i: str, bt_j: str) -> dict:
    p = opls['bond'].get((bt_i, bt_j))
    if p:
        return p
    ei, ej = _element_of(bt_i), _element_of(bt_j)
    key = tuple(sorted([ei, ej]))
    fb = _BOND_DEFAULTS.get(key) or _BOND_DEFAULTS.get((key[1], key[0]))
    if fb is None:
        warnings.warn(f"Bond {bt_i}-{bt_j} not found; using r0=1.5 k=300")
        fb = {'r0': 1.5, 'k': 300.0}
    else:
        warnings.warn(f"Bond {bt_i}-{bt_j} not in OPLS file; using element-default")
    return fb


def lookup_angle(opls: dict, bt_i: str, bt_j: str, bt_k: str) -> dict:
    p = opls['angle'].get((bt_i, bt_j, bt_k))
    if p:
        return p
    ei, ej, ek = _element_of(bt_i), _element_of(bt_j), _element_of(bt_k)
    fb = _ANGLE_DEFAULTS.get((ei, ej, ek)) or _ANGLE_DEFAULTS.get((ek, ej, ei))
    if fb is None:
        warnings.warn(f"Angle {bt_i}-{bt_j}-{bt_k} not found; using 109.5° k=50")
        fb = {'th0': 109.5, 'k': 50.0}
    else:
        warnings.warn(f"Angle {bt_i}-{bt_j}-{bt_k} not in OPLS file; using element-default")
    return fb


def lookup_dihedral(opls: dict, bt_i: str, bt_j: str, bt_k: str, bt_l: str) -> list:
    p = opls['dihedral'].get((bt_i, bt_j, bt_k, bt_l))
    if p:
        return p
    ei, ej, ek, el = (_element_of(bt_i), _element_of(bt_j),
                      _element_of(bt_k), _element_of(bt_l))
    fb = _DIHEDRAL_DEFAULTS.get((ei, ej, ek, el)) or _DIHEDRAL_DEFAULTS.get((el, ek, ej, ei))
    if fb is None:
        warnings.warn(f"Dihedral {bt_i}-{bt_j}-{bt_k}-{bt_l} not found; using zeros")
        fb = [0.0, 0.0, 0.0, 0.0]
    else:
        warnings.warn(f"Dihedral {bt_i}-{bt_j}-{bt_k}-{bt_l} not in OPLS file; using element-default")
    return fb


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE GEOMETRY  –  bond-graph BFS traversal engine
# ─────────────────────────────────────────────────────────────────────────────

def _perp_frame(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to unit vector v."""
    ref  = np.array([1., 0., 0.]) if abs(v[0]) < 0.9 else np.array([0., 1., 0.])
    perp = np.cross(v, ref);  perp  /= np.linalg.norm(perp)
    perp2 = np.cross(v, perp)
    return perp, perp2


def place_by_zmatrix(pos_gp: np.ndarray | None,
                     pos_p:  np.ndarray,
                     bond_length: float,
                     angle_deg:   float,
                     torsion:     float) -> np.ndarray:
    """
    Place one atom using internal (Z-matrix) coordinates.

    Parameters
    ----------
    pos_gp      : grandparent position – defines the torsion reference plane.
                  If None, the torsion angle is used to orient around a fixed axis
                  so that sibling atoms get different positions.
    pos_p       : parent position
    bond_length : distance p→new  (Å)
    angle_deg   : valence angle gp-p-new  (degrees)
    torsion     : dihedral angle around gp-p bond  (radians)
    """
    if pos_gp is None:
        # No grandparent: place using torsion to rotate around the z-axis
        # so that sibling atoms (same parent, no grandparent) get spread out.
        theta = np.radians(180.0 - angle_deg)   # deflection from z-axis
        ref   = np.array([1., 0., 0.])
        perp2 = np.array([0., 1., 0.])
        axis  = np.array([0., 0., 1.])
        direction = (np.cos(theta) * axis
                     + np.sin(theta) * (np.cos(torsion) * ref
                                        + np.sin(torsion) * perp2))
        direction /= np.linalg.norm(direction)
        return pos_p + bond_length * direction

    v = pos_p - pos_gp
    v /= np.linalg.norm(v)

    deflection = np.radians(180.0 - angle_deg)
    perp, perp2 = _perp_frame(v)

    new_dir = (np.cos(deflection) * v
               + np.sin(deflection) * (np.cos(torsion) * perp
                                       + np.sin(torsion) * perp2))
    new_dir /= np.linalg.norm(new_dir)
    return pos_p + bond_length * new_dir


def find_rings(adj: dict[int, set]) -> list[list[int]]:
    """
    Find all smallest rings (3-8 members) in the bond graph using back-edge BFS.

    For every bond (u, v) we run BFS from u excluding that direct edge.  If v is
    reachable the shortest path u→v + the direct bond u-v forms a ring.  Duplicate
    ring *sets* are suppressed.

    Returns list of rings, each ring is an ordered list of local_ids.
    """
    rings: list[list[int]] = []
    seen_sets: list[frozenset] = []

    all_nodes = sorted(adj.keys())
    bonds: list[tuple[int, int]] = []
    for u in all_nodes:
        for v in adj[u]:
            if u < v:
                bonds.append((u, v))

    for (u, v) in bonds:
        # BFS from u to v, forbidden: direct edge u-v
        prev: dict[int, int | None] = {u: None}
        q = deque([u])
        found = False
        while q and not found:
            cur = q.popleft()
            for nxt in adj[cur]:
                if cur == u and nxt == v:
                    continue      # skip direct bond
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == v:
                    found = True
                    break
                q.append(nxt)

        if not found:
            continue

        # Reconstruct ring: v → ... → u  +  direct u-v
        path: list[int] = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = prev[cur]       # type: ignore[assignment]
        # path = [v, ..., u];  ring = path (includes both endpoints)

        if len(path) < 3 or len(path) > 8:
            continue
        rs = frozenset(path)
        if rs not in seen_sets:
            seen_sets.append(rs)
            rings.append(path)    # ordered: v → ... → u

    return rings


def place_ring_atoms(ring_lids:     list[int],
                     attach_lid:    int,
                     attach_pos:    np.ndarray,
                     parent_pos:    np.ndarray,
                     local_info:    dict,
                     opls:          dict,
                     rng:           np.random.Generator) -> dict[int, np.ndarray]:
    """
    Place all atoms in one ring as a regular polygon.

    The ring atom `attach_lid` (already placed at `attach_pos`) serves as the
    origin.  `parent_pos` is the already-placed atom outside the ring that is
    bonded to `attach_lid`; it defines the in-plane orientation.

    The ring is placed in a plane that is randomly rotated around the
    attach_lid→parent_pos bond axis (amorphous packing).

    Returns {local_id: position} for ALL ring atoms including attach_lid.
    """
    n = len(ring_lids)
    if n < 3:
        return {attach_lid: attach_pos}

    # ── average ring bond length ──────────────────────────────────────────
    avg_bl = 0.0
    cnt = 0
    for k in range(n):
        bti = local_info[ring_lids[k]]['bond_type']
        btj = local_info[ring_lids[(k + 1) % n]]['bond_type']
        avg_bl += lookup_bond(opls, bti, btj)['r0']
        cnt += 1
    avg_bl /= cnt

    # ── ring radius for regular n-gon ─────────────────────────────────────
    R = avg_bl / (2.0 * np.sin(np.pi / n))

    # ── coordinate frame ──────────────────────────────────────────────────
    # bond_dir: unit vector from parent → attach (direction the ring "grows" from)
    bond_dir = attach_pos - parent_pos
    bond_dir /= np.linalg.norm(bond_dir)

    # ring_normal: perpendicular to bond_dir, random orientation for amorphous packing
    rand_vec = rng.normal(size=3)
    ring_normal = rand_vec - np.dot(rand_vec, bond_dir) * bond_dir
    if np.linalg.norm(ring_normal) < 1e-8:
        ring_normal = np.cross(bond_dir, np.array([0., 0., 1.]))
    ring_normal /= np.linalg.norm(ring_normal)

    # in_plane_dir: from ring center toward attach_lid (atom 0 of the polygon)
    # We want attach_lid to sit at ring center + R * in_plane_dir
    # in_plane_dir must be perpendicular to ring_normal; choose it aligned with bond_dir
    in_plane_dir = bond_dir - np.dot(bond_dir, ring_normal) * ring_normal
    if np.linalg.norm(in_plane_dir) < 1e-8:
        in_plane_dir, _ = _perp_frame(ring_normal)
    else:
        in_plane_dir /= np.linalg.norm(in_plane_dir)

    perp_in_plane = np.cross(ring_normal, in_plane_dir)

    # ring center
    ring_center = attach_pos - R * in_plane_dir

    # ── place atoms around the polygon ────────────────────────────────────
    # Reorder ring so that attach_lid is first
    try:
        start = ring_lids.index(attach_lid)
    except ValueError:
        start = 0
    ordered = ring_lids[start:] + ring_lids[:start]

    positions: dict[int, np.ndarray] = {}
    for k, lid in enumerate(ordered):
        angle = 2.0 * np.pi * k / n
        pos = (ring_center
               + R * np.cos(angle) * in_plane_dir
               + R * np.sin(angle) * perp_in_plane)
        positions[lid] = pos

    return positions


class _SpatialGrid:
    """
    Simple spatial hash grid for fast nearest-neighbour queries.
    Avoids O(N²) distance checks when N is large.
    """
    __slots__ = ('cell', 'positions', 'inv_cell')

    def __init__(self, cell_size: float = 3.0):
        self.cell      = cell_size
        self.inv_cell  = 1.0 / cell_size
        self.positions: dict[tuple, list[np.ndarray]] = {}

    def _key(self, pos: np.ndarray) -> tuple:
        return (int(pos[0] * self.inv_cell),
                int(pos[1] * self.inv_cell),
                int(pos[2] * self.inv_cell))

    def add(self, pos: np.ndarray) -> None:
        self.positions.setdefault(self._key(pos), []).append(pos)

    def has_clash(self, pos: np.ndarray, min_dist: float) -> bool:
        ki, kj, kk = self._key(pos)
        r = int(np.ceil(min_dist * self.inv_cell))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    nbrs = self.positions.get((ki+di, kj+dj, kk+dk))
                    if nbrs:
                        for nb in nbrs:
                            if np.linalg.norm(nb - pos) < min_dist:
                                return True
        return False


def _has_clash(pos: np.ndarray,
               existing: list[np.ndarray],
               min_dist: float = 2.0,
               is_H: bool = False,
               existing_is_H: list = None) -> bool:
    """Fallback O(N) clash check for small lists.

    Element-aware: the relaxed H threshold (natural H-H contacts in a CH2/CH3
    are ~1.76 A and must not be flagged) is applied ONLY to H-H pairs. For an
    H against a HEAVY atom a stricter threshold (>=0.9 A) is used, so a hydrogen
    cannot be accepted overlapping a carbon/oxygen (e.g. the 0.72 A C-H overlap
    seen in poly(acrylic acid) side chains). Without element info it matches the
    old single-threshold behaviour.
    """
    if not existing:
        return False
    arr = np.array(existing)
    dists = np.linalg.norm(arr - pos, axis=1)
    if is_H and existing_is_H is not None:
        eh = np.array(existing_is_H, dtype=bool)
        h_heavy_thr = max(min_dist, 0.9)
        thr = np.where(eh, min_dist, h_heavy_thr)
        return bool(np.any(dists < thr))
    return bool(np.any(dists < min_dist))


def place_monomer(intra_bonds:     list[list[int]],
                  local_info:      dict[int, dict],
                  opls:            dict,
                  head_lid:        int,
                  anchor_pos:      np.ndarray,
                  prev_tail_pos:   np.ndarray | None,
                  rng:             np.random.Generator,
                  grid:            '_SpatialGrid | None' = None,
                  n_torsion_tries: int = 500,
                  min_clash_dist:  float = 1.0) -> tuple[dict[int, np.ndarray], int]:
    """
    Place ALL atoms in one repeat unit via BFS on the intra-monomer bond graph.
    For each non-ring atom, try up to n_torsion_tries random torsion angles and
    accept the first one that does not clash with existing_pos.

    Parameters
    ----------
    intra_bonds     : [[i,j], ...] local_id pairs for bonds inside this monomer
    local_info      : {local_id: opls_info dict}
    opls            : parsed OPLS-AA tables
    head_lid        : local_id of the atom already placed at anchor_pos
    anchor_pos      : 3-D position of head_lid
    prev_tail_pos   : position of tail atom of the previous monomer (grandparent)
    rng             : numpy random generator
    grid            : _SpatialGrid instance for fast cross-chain clash checking
    n_torsion_tries : number of torsion angle attempts before giving up (default 200)
    min_clash_dist  : minimum allowed distance between non-bonded atoms (Å, default 2.0)

    Returns
    -------
    tuple of ({local_id: np.ndarray}, fallback_count)
      fallback_count: number of atoms that could not find a clash-free position
    """
    if grid is None:
        grid = _SpatialGrid()

    fallback_count: int = 0  # number of atoms that needed fallback placement

    # ── build adjacency ───────────────────────────────────────────────────
    adj: dict[int, set[int]] = defaultdict(set)
    for (i, j) in intra_bonds:
        adj[i].add(j)
        adj[j].add(i)

    # ── detect rings ──────────────────────────────────────────────────────
    rings = find_rings(adj)
    ring_atom_to_ring: dict[int, list[int]] = {}
    for ring in rings:
        for lid in ring:
            ring_atom_to_ring[lid] = ring

    placed: dict[int, np.ndarray]     = {head_lid: anchor_pos}
    bfs_parent: dict[int, int | None] = {head_lid: None}
    visited: set[int]                 = {head_lid}
    placed_rings: set[frozenset]      = set()

    # Running list of positions placed so far in this monomer + existing
    local_placed: list[np.ndarray] = [anchor_pos]
    local_is_H: list = [local_info[head_lid]['bond_type'].startswith('H')]

    queue: deque[int] = deque([head_lid])

    while queue:
        cur = queue.popleft()
        cur_pos = placed[cur]

        par = bfs_parent[cur]
        if par is not None:
            gp_pos: np.ndarray | None = placed[par]
        else:
            gp_pos = prev_tail_pos

        cur_bt = local_info[cur]['bond_type']

        for nxt in sorted(adj[cur]):
            if nxt in visited:
                continue

            nxt_bt = local_info[nxt]['bond_type']

            # ── ring atom ────────────────────────────────────────────────
            if nxt in ring_atom_to_ring:
                ring = ring_atom_to_ring[nxt]
                ring_key = frozenset(ring)

                if ring_key in placed_rings:
                    for lid in ring:
                        if lid not in visited:
                            visited.add(lid)
                            bfs_parent[lid] = cur
                            queue.append(lid)
                    continue

                ring_positions = place_ring_atoms(
                    ring_lids=ring,
                    attach_lid=nxt,
                    attach_pos=place_by_zmatrix(
                        gp_pos, cur_pos,
                        lookup_bond(opls, cur_bt, nxt_bt)['r0'],
                        lookup_angle(opls,
                                     local_info[par]['bond_type'] if par else cur_bt,
                                     cur_bt, nxt_bt)['th0'] if gp_pos is not None else 120.0,
                        rng.uniform(0, 2 * np.pi)
                    ),
                    parent_pos=cur_pos,
                    local_info=local_info,
                    opls=opls,
                    rng=rng,
                )
                for lid, pos in ring_positions.items():
                    placed[lid] = pos
                    local_placed.append(pos)
                    local_is_H.append(local_info[lid]['bond_type'].startswith('H'))
                    if lid not in visited:
                        visited.add(lid)
                        bfs_parent[lid] = cur
                        queue.append(lid)
                placed_rings.add(ring_key)
                continue

            # ── normal (non-ring) atom — clash-aware torsion retry ────────
            bl  = lookup_bond(opls, cur_bt, nxt_bt)['r0']
            ang = (lookup_angle(opls,
                                local_info[par]['bond_type'] if par else cur_bt,
                                cur_bt, nxt_bt)['th0']
                   if gp_pos is not None else 109.5)

            best_pos = None
            fallback_candidates: list[np.ndarray] = []

            for _ in range(n_torsion_tries):
                torsion   = rng.uniform(0, 2 * np.pi)
                candidate = place_by_zmatrix(gp_pos, cur_pos, bl, ang, torsion)
                # Smaller threshold for H atoms (natural H-H in CH2 ≈ 1.76 Å)
                is_H      = local_info[nxt]['bond_type'].startswith('H')
                clash_thr = 0.7 if is_H else min_clash_dist
                # Check against intra-monomer atoms (fast list) + grid (cross-chain)
                if (not _has_clash(candidate, local_placed, clash_thr,
                                   is_H=is_H, existing_is_H=local_is_H) and
                        not grid.has_clash(candidate, clash_thr)):
                    best_pos = candidate
                    break
                fallback_candidates.append(candidate)

            # Fallback: pick the candidate with maximum minimum-distance to
            # already-placed atoms (least-clashing position) instead of random.
            if best_pos is None:
                if fallback_candidates and local_placed:
                    arr = np.array(local_placed)
                    min_dists = [float(np.min(np.linalg.norm(arr - c, axis=1)))
                                 for c in fallback_candidates]
                    best_pos = fallback_candidates[int(np.argmax(min_dists))]
                elif fallback_candidates:
                    best_pos = fallback_candidates[-1]
                else:
                    best_pos = place_by_zmatrix(gp_pos, cur_pos, bl, ang,
                                                rng.uniform(0, 2 * np.pi))
                fallback_count += 1

            placed[nxt] = best_pos
            local_placed.append(best_pos)
            local_is_H.append(local_info[nxt]['bond_type'].startswith('H'))
            visited.add(nxt)
            bfs_parent[nxt] = cur
            queue.append(nxt)

    # ── warn about any unplaced atoms (disconnected graph) ────────────────
    all_lids = set(local_info.keys())
    unplaced = all_lids - set(placed.keys())
    if unplaced:
        warnings.warn(
            f"Atoms {sorted(unplaced)} unreachable from head_atom={head_lid}. "
            "Check that the monomer bond graph is fully connected."
        )
        for lid in sorted(unplaced):
            placed[lid] = anchor_pos + rng.uniform(-0.5, 0.5, size=3)

    return placed, fallback_count


# ─────────────────────────────────────────────────────────────────────────────
# BOX SIZE FROM DENSITY
# ─────────────────────────────────────────────────────────────────────────────

AVOGADRO = 6.02214076e23  # mol⁻¹
CM3_TO_A3 = 1.0e24         # 1 cm³ = 1e24 Å³


def calculate_box_size(n_chains: int, n_monomers: int,
                        monomer_mass_g_per_mol: float,
                        density_g_per_cm3: float = 0.9) -> float:
    """Return cubic simulation box side length in Å."""
    total_mass = n_chains * n_monomers * monomer_mass_g_per_mol  # g/mol
    volume_cm3 = total_mass / (density_g_per_cm3 * AVOGADRO)     # cm³
    volume_A3  = volume_cm3 * CM3_TO_A3
    return volume_A3 ** (1.0 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY ENUMERATION (angles & dihedrals from bond graph)
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency(bonds: list[tuple[int, int]]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for (a, b) in bonds:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def enumerate_angles(bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Return list of unique (i, j, k) triples sharing atom j."""
    adj = build_adjacency(bonds)
    seen = set()
    result = []
    for j, neighbors in adj.items():
        nl = sorted(neighbors)
        for idx_a in range(len(nl)):
            for idx_b in range(idx_a + 1, len(nl)):
                i, k = nl[idx_a], nl[idx_b]
                key = (min(i, k), j, max(i, k))
                if key not in seen:
                    seen.add(key)
                    result.append((i, j, k))
    return result


def enumerate_dihedrals(bonds: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Return list of unique (i, j, k, l) quadruples about bond j-k."""
    adj = build_adjacency(bonds)
    seen = set()
    result = []
    for (j, k) in bonds:
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                key = tuple(sorted([(i, j, k, l), (l, k, j, i)])[0])
                if key not in seen:
                    seen.add(key)
                    result.append((i, j, k, l))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SYSTEM BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_polymer_system(poly_def: dict, opls: dict,
                          n_chains: int = 20, n_monomers: int = 10,
                          density: float = 0.9,
                          seed: int | None = None) -> dict:
    """
    Build the full amorphous polymer system from a poly_def dict.

    Coordinate algorithm
    ────────────────────
    Each monomer is placed by BFS traversal of the intra-monomer bond graph
    (place_monomer).  The first atom of each monomer is connected to the tail
    atom of the previous monomer, which provides the grandparent reference for
    z-matrix placement.  The chain grows monomer by monomer; a random rigid-body
    offset is applied to each chain to fill the simulation box.

    Returns a system dict ready for write_lammps_datafile().
    """
    rng = np.random.default_rng(seed)

    # ── 1. Resolve OPLS info for every atom in the monomer ───────────────────
    monomer_atoms = poly_def['monomer_atoms']
    n_mono_atoms  = len(monomer_atoms)
    head_id       = poly_def['head_atom']
    tail_id       = poly_def['tail_atom']
    intra_bonds   = [list(b) for b in poly_def['intra_bonds']]

    opls_atoms = opls['atom']

    local_info: dict[int, dict] = {}
    for a in monomer_atoms:
        lid = a['local_id']
        oid = a['opls_id']
        if oid not in opls_atoms:
            raise ValueError(f"opls_id '{oid}' not found in atom.txt")
        odata = opls_atoms[oid].copy()
        # Only override OPLS charge if poly_def explicitly provides a non-zero value
        if 'charge' in a and a['charge'] != 0.0:
            odata['charge'] = float(a['charge'])
        # else: keep OPLS default charge
        local_info[lid] = odata

    # Unique atom types ordered by first appearance
    seen_types: list[str] = []
    for a in monomer_atoms:
        oid = a['opls_id']
        if oid not in seen_types:
            seen_types.append(oid)
    opls_id_to_type_id = {oid: i + 1 for i, oid in enumerate(seen_types)}

    # ── 2. Monomer mass → box size ────────────────────────────────────────────
    monomer_mass = sum(local_info[a['local_id']]['mass'] for a in monomer_atoms)
    # Use target density directly so the box size matches what the user confirmed.
    # Initial overlaps are resolved by the minimization step in the in. file.
    box_L = calculate_box_size(n_chains, n_monomers, monomer_mass, density)

    # ── 3. Global atom ID helper ──────────────────────────────────────────────
    def global_id(chain_i: int, mono_i: int, local_i: int) -> int:
        return (chain_i * n_monomers * n_mono_atoms
                + mono_i * n_mono_atoms
                + (local_i - 1) + 1)

    all_atoms:     list = []   # (gid, mol_id, type_id, charge, x, y, z)
    all_bonds:     list = []   # (gid_i, gid_j)
    all_angles:    list = []
    all_dihedrals: list = []

    # ── 4. Build every chain ──────────────────────────────────────────────────
    # Grid-based origin placement: divide box into n_chains cells so chains
    # start well-separated. A small random jitter prevents perfect alignment.
    grid_n    = int(np.ceil(n_chains ** (1.0 / 3.0)))
    cell_size = box_L / grid_n
    grid_pts  = [(i, j, k)
                 for i in range(grid_n)
                 for j in range(grid_n)
                 for k in range(grid_n)]
    rng.shuffle(grid_pts)

    all_placed_positions: list[np.ndarray] = []   # for cross-chain clash reference
    clash_grid = _SpatialGrid(cell_size=3.0)       # fast spatial lookup

    # ── Chain retry constants ─────────────────────────────────────────────────
    MAX_CHAIN_RETRIES   = 5
    # Fraction of atoms allowed to fall back before triggering a chain retry
    _FALLBACK_LIMIT = max(1, n_monomers * n_mono_atoms // 5)   # ~20 %
    # Jitter grows with each retry: 0.20 → 0.35 → 0.50 (capped)
    def _jitter(retry: int) -> float:
        return min(0.20 + 0.15 * retry, 0.50)

    for chain_i in range(n_chains):
        mol_id = chain_i + 1

        gi, gj, gk = grid_pts[chain_i % len(grid_pts)]

        accepted_atoms:     list = []   # filled once the chain is accepted
        accepted_positions: list[np.ndarray] = []

        for retry in range(MAX_CHAIN_RETRIES + 1):
            jit = _jitter(retry)

            # Origin: grid cell centre ± jitter
            origin = (np.array([gi, gj, gk], dtype=float) + 0.5) * cell_size
            origin += rng.uniform(-jit * cell_size, jit * cell_size, size=3)
            origin = np.clip(origin, 0.0, box_L)

            anchor_pos    = origin.copy()
            prev_tail_pos: np.ndarray | None = None

            chain_atom_positions: dict[tuple[int, int], np.ndarray] = {}

            chain_atoms_tmp:     list = []
            chain_positions_tmp: list[np.ndarray] = []
            total_fallbacks      = 0

            # ── intra-chain grid: accumulates placed monomers of THIS chain ──
            intra_chain_grid = _SpatialGrid(cell_size=3.0)

            class _CombinedGrid:
                def __init__(self, g1, g2):
                    self._g1, self._g2 = g1, g2
                def has_clash(self, pos, min_dist):
                    return (self._g1.has_clash(pos, min_dist) or
                            self._g2.has_clash(pos, min_dist))

            combined_grid = _CombinedGrid(clash_grid, intra_chain_grid)

            # ── Check origin itself against cross-chain grid ──────────────
            # Without this, the first atom of a new chain can land on top
            # of an already-placed chain atom (inter-chain overlap).
            _ORIGIN_TRIES = 50
            for _ot in range(_ORIGIN_TRIES):
                if not clash_grid.has_clash(origin, 1.0):
                    break
                origin = (np.array([gi, gj, gk], dtype=float) + 0.5) * cell_size
                origin += rng.uniform(-jit * cell_size, jit * cell_size, size=3)
                origin = np.clip(origin, 0.0, box_L)

            anchor_pos    = origin.copy()
            prev_tail_pos: np.ndarray | None = None

            for mono_i in range(n_monomers):
                local_positions, n_fb = place_monomer(
                    intra_bonds   = intra_bonds,
                    local_info    = local_info,
                    opls          = opls,
                    head_lid      = head_id,
                    anchor_pos    = anchor_pos,
                    prev_tail_pos = prev_tail_pos,
                    rng           = rng,
                    grid          = combined_grid,
                )
                total_fallbacks += n_fb

                # Record atom positions (temporary until chain is accepted)
                for lid, pos in local_positions.items():
                    chain_atom_positions[(mono_i, lid)] = pos
                    gid  = global_id(chain_i, mono_i, lid)
                    info = local_info[lid]
                    chain_atoms_tmp.append((gid, mol_id,
                                            opls_id_to_type_id[info['opls_id']],
                                            info['charge'],
                                            pos[0], pos[1], pos[2]))
                    chain_positions_tmp.append(pos)

                # Add this monomer's atoms to intra-chain grid immediately
                for pos in local_positions.values():
                    intra_chain_grid.add(pos)

                # ── Prepare anchor for next monomer ──────────────────────────
                tail_pos = local_positions[tail_id]
                head_pos = local_positions[head_id]
                bt_tail  = local_info[tail_id]['bond_type']
                bt_head  = local_info[head_id]['bond_type']
                inter_bl = lookup_bond(opls, bt_tail, bt_head)['r0']

                _ANCHOR_TRIES = 200
                _anchor_candidates = []
                # Current chain bounding box (unwrapped)
                _cur_pts   = np.array(chain_positions_tmp) if chain_positions_tmp else np.zeros((1,3))
                _cur_min   = _cur_pts.min(axis=0)
                _cur_max   = _cur_pts.max(axis=0)
                _half      = box_L / 2.0

                for _at in range(_ANCHOR_TRIES):
                    gd = tail_pos - head_pos
                    norm = np.linalg.norm(gd)
                    gd = gd / norm if norm > 1e-8 else rng.normal(size=3) / np.linalg.norm(rng.normal(size=3))

                    _deflect = np.radians(180.0 - 109.5)
                    _p1, _p2 = _perp_frame(gd)

                    # If current extent is already >= 80% of L/2, force gauche to fold chain
                    _cur_extent = float(np.max(_cur_max - _cur_min))
                    if _cur_extent >= 0.80 * _half:
                        _tau = rng.choice([np.pi / 3.0, -np.pi / 3.0])  # gauche only
                    else:
                        _tau = rng.choice([np.pi, np.pi / 3.0, -np.pi / 3.0],
                                           p=[0.50, 0.25, 0.25])

                    gd = (np.cos(_deflect) * gd
                          + np.sin(_deflect) * (np.cos(_tau) * _p1
                                                + np.sin(_tau) * _p2))
                    gd /= np.linalg.norm(gd)

                    candidate_anchor = tail_pos + inter_bl * gd

                    # Reject if new anchor would push extent over L/2
                    new_min = np.minimum(_cur_min, candidate_anchor)
                    new_max = np.maximum(_cur_max, candidate_anchor)
                    new_ext = float(np.max(new_max - new_min))
                    if new_ext > _half:
                        continue  # try another torsion angle

                    if not combined_grid.has_clash(candidate_anchor, 1.0):
                        anchor_pos = candidate_anchor
                        break
                    _anchor_candidates.append(candidate_anchor)
                else:
                    # No ideal anchor: pick least-extent candidate
                    if _anchor_candidates:
                        anchor_pos = _anchor_candidates[-1]
                    else:
                        anchor_pos = tail_pos + inter_bl * gd

                prev_tail_pos = tail_pos

            # ── Accept or retry ───────────────────────────────────────────────
            # Also reject if chain extent > box_L/2: LAMMPS MPI would lose
            # dihedral atoms across proc boundaries ("Dihedral atoms missing").
            if chain_positions_tmp:
                _pts   = np.array(chain_positions_tmp)
                _ext   = float(np.max(np.max(_pts, axis=0) - np.min(_pts, axis=0)))
            else:
                _ext = 0.0
            _ok_fb  = total_fallbacks <= _FALLBACK_LIMIT
            _ok_ext = _ext <= box_L / 2.0

            if (_ok_fb and _ok_ext) or retry == MAX_CHAIN_RETRIES:
                accepted_atoms     = chain_atoms_tmp
                accepted_positions = chain_positions_tmp
                if not _ok_fb:
                    warnings.warn(
                        f"Chain {mol_id}: {total_fallbacks} fallback placements "
                        f"after {MAX_CHAIN_RETRIES} retries – accepted anyway."
                    )
                if not _ok_ext:
                    warnings.warn(
                        f"Chain {mol_id}: extent {_ext:.1f} Å > L/2 {box_L/2:.1f} Å "
                        f"after {MAX_CHAIN_RETRIES} retries – accepted anyway."
                    )
                break
            else:
                if not _ok_ext:
                    warnings.warn(
                        f"Chain {mol_id}: extent {_ext:.1f} Å > L/2 {box_L/2:.1f} Å "
                        f"(retry {retry+1}/{MAX_CHAIN_RETRIES})"
                    )
                else:
                    warnings.warn(
                        f"Chain {mol_id}: {total_fallbacks} fallbacks "
                        f"(retry {retry + 1}/{MAX_CHAIN_RETRIES}, jitter={jit:.2f})"
                    )

        # ── Commit accepted chain ─────────────────────────────────────────────
        for atom in accepted_atoms:
            all_atoms.append(atom)
        for pos in accepted_positions:
            all_placed_positions.append(pos)
            clash_grid.add(pos)

        # ── Bonds ─────────────────────────────────────────────────────────────
        for mono_i in range(n_monomers):
            for (li, lj) in intra_bonds:
                all_bonds.append((global_id(chain_i, mono_i, li),
                                  global_id(chain_i, mono_i, lj)))
            if mono_i < n_monomers - 1:
                all_bonds.append((global_id(chain_i, mono_i,     tail_id),
                                  global_id(chain_i, mono_i + 1, head_id)))

        # ── Angles & Dihedrals from this chain's bond graph ───────────────────
        n_chain_bonds = n_monomers * len(intra_bonds) + (n_monomers - 1)
        chain_bonds = all_bonds[-n_chain_bonds:]
        for triple in enumerate_angles(chain_bonds):
            all_angles.append(triple)
        for quad in enumerate_dihedrals(chain_bonds):
            all_dihedrals.append(quad)

    # ── 5. Assign unique force-field types to bonds / angles / dihedrals ─────
    gid_to_bt: dict[int, str] = {}
    for (gid, mol_id, type_id, charge, x, y, z) in all_atoms:
        gid_to_bt[gid] = opls_atoms[seen_types[type_id - 1]]['bond_type']

    def _type_id(keys, type_map, norm_key):
        rev_key = norm_key[::-1]
        if norm_key in type_map:
            return type_map[norm_key]
        if rev_key in type_map:
            return type_map[rev_key]
        tid = len(keys) + 1
        type_map[norm_key] = tid
        keys.append(norm_key)
        return tid

    bond_type_keys: list = [];  bond_type_map: dict = {}
    typed_bonds = []
    for (ai, aj) in all_bonds:
        key = (gid_to_bt[ai], gid_to_bt[aj])
        typed_bonds.append((_type_id(bond_type_keys, bond_type_map, key), ai, aj))

    angle_type_keys: list = [];  angle_type_map: dict = {}
    typed_angles = []
    for (ai, aj, ak) in all_angles:
        if not all(g in gid_to_bt for g in (ai, aj, ak)):
            continue
        key = (gid_to_bt[ai], gid_to_bt[aj], gid_to_bt[ak])
        typed_angles.append((_type_id(angle_type_keys, angle_type_map, key), ai, aj, ak))

    dih_type_keys: list = [];  dih_type_map: dict = {}
    typed_dihedrals = []
    for (ai, aj, ak, al) in all_dihedrals:
        if not all(g in gid_to_bt for g in (ai, aj, ak, al)):
            continue
        key = (gid_to_bt[ai], gid_to_bt[aj], gid_to_bt[ak], gid_to_bt[al])
        typed_dihedrals.append((_type_id(dih_type_keys, dih_type_map, key), ai, aj, ak, al))

    return {
        'poly_def':        poly_def,
        'n_chains':        n_chains,
        'n_monomers':      n_monomers,
        'density':         density,
        'box_L':           box_L,
        'monomer_mass':    monomer_mass,
        'atom_type_ids':   opls_id_to_type_id,
        'seen_types':      seen_types,
        'bond_type_keys':  bond_type_keys,
        'angle_type_keys': angle_type_keys,
        'dih_type_keys':   dih_type_keys,
        'atoms':           all_atoms,
        'bonds':           typed_bonds,
        'angles':          typed_angles,
        'dihedrals':       typed_dihedrals,
        'gid_to_bt':       gid_to_bt,
        'opls_atoms':      opls['atom'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAMMPS DATA FILE WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_lammps_datafile(system: dict, opls: dict, output_path: str):
    """
    Write a LAMMPS data file in 'full' atom style with:
      Masses, Pair Coeffs, Bond Coeffs, Angle Coeffs, Dihedral Coeffs,
      Atoms, Bonds, Angles, Dihedrals
    """
    pdef   = system['poly_def']
    L      = system['box_L']
    atoms  = sorted(system['atoms'],  key=lambda x: x[0])
    bonds  = system['bonds']
    angles = system['angles']
    dihedrals = system['dihedrals']

    n_atom_types = len(system['seen_types'])
    n_bond_types = len(system['bond_type_keys'])
    n_angle_types = len(system['angle_type_keys'])
    n_dih_types   = len(system['dih_type_keys'])

    opls_atoms = opls['atom']

    with open(output_path, 'w') as f:
        # ── Header ────────────────────────────────────────────────────────────
        f.write(f"LAMMPS data file – {pdef.get('polymer_name','polymer')} "
                f"({system['n_chains']} chains × {system['n_monomers']} monomers, "
                f"OPLS-AA, density {system['density']:.2f} g/cm³)\n\n")

        f.write(f"   {len(atoms)} atoms\n")
        f.write(f"   {len(bonds)} bonds\n")
        f.write(f"   {len(angles)} angles\n")
        f.write(f"   {len(dihedrals)} dihedrals\n\n")

        f.write(f"   {n_atom_types} atom types\n")
        f.write(f"   {n_bond_types} bond types\n")
        f.write(f"   {n_angle_types} angle types\n")
        f.write(f"   {n_dih_types} dihedral types\n\n")

        f.write(f"     0.000000000    {L:16.9f} xlo xhi\n")
        f.write(f"     0.000000000    {L:16.9f} ylo yhi\n")
        f.write(f"     0.000000000    {L:16.9f} zlo zhi\n")

        # ── Masses ────────────────────────────────────────────────────────────
        f.write("\nMasses\n\n")
        for i, oid in enumerate(system['seen_types'], 1):
            info = opls_atoms[oid]
            f.write(f"   {i}  {info['mass']:.6f} # {info['bond_type']}  ({oid})\n")

        # ── Pair Coeffs ───────────────────────────────────────────────────────
        f.write("\nPair Coeffs # lj/cut/coul/long\n\n")
        for i, oid in enumerate(system['seen_types'], 1):
            info = opls_atoms[oid]
            f.write(f"   {i}   {info['epsilon']:.10f}   {info['sigma']:.10f}"
                    f" # {info['bond_type']}  ({oid})\n")

        # ── Bond Coeffs ───────────────────────────────────────────────────────
        f.write("\nBond Coeffs # harmonic\n\n")
        for i, (bti, btj) in enumerate(system['bond_type_keys'], 1):
            p = lookup_bond(opls, bti, btj)
            f.write(f"   {i}   {p['k']:.4f}     {p['r0']:.4f}"
                    f" # {bti}-{btj}\n")

        # ── Angle Coeffs ──────────────────────────────────────────────────────
        f.write("\nAngle Coeffs # harmonic\n\n")
        for i, (bti, btj, btk) in enumerate(system['angle_type_keys'], 1):
            p = lookup_angle(opls, bti, btj, btk)
            f.write(f"   {i}    {p['k']:.4f}   {p['th0']:.4f}"
                    f" # {bti}-{btj}-{btk}\n")

        # ── Dihedral Coeffs ───────────────────────────────────────────────────
        f.write("\nDihedral Coeffs # opls\n\n")
        for i, (bti, btj, btk, btl) in enumerate(system['dih_type_keys'], 1):
            k = lookup_dihedral(opls, bti, btj, btk, btl)
            f.write(f"  {i}     {k[0]:.6f}   {k[1]:.6f}  {k[2]:.6f}   {k[3]:.6f}"
                    f" # {bti}-{btj}-{btk}-{btl}\n")

        # ── Atoms # full ──────────────────────────────────────────────────────
        f.write("\nAtoms # full\n\n")
        for (gid, mol_id, type_id, charge, x, y, z) in atoms:
            oid = system['seen_types'][type_id - 1]
            bt  = opls_atoms[oid]['bond_type']
            # Wrap into [0, L) and record image flags.
            # Atoms built via random walk can lie outside [0, box_L].
            # With image=0,0,0 LAMMPS sees them as outside its MPI domain →
            # "Bond/Dihedral atoms missing". Correct image flags let LAMMPS
            # reconstruct unwrapped pos: x_real = wx + ix * L.
            ix_ = int(np.floor(x / L));  wx = x - ix_ * L
            iy_ = int(np.floor(y / L));  wy = y - iy_ * L
            iz_ = int(np.floor(z / L));  wz = z - iz_ * L
            f.write(f"      {gid}      {mol_id}   {type_id}  {charge:10.6f}"
                    f"    {wx:16.9f}    {wy:16.9f}    {wz:16.9f}"
                    f"   {ix_}   {iy_}   {iz_} # {bt}\n")

        # ── Bonds ─────────────────────────────────────────────────────────────
        f.write("\nBonds\n\n")
        for bond_id, (tid, ai, aj) in enumerate(bonds, 1):
            f.write(f"     {bond_id}   {tid}      {ai}      {aj}\n")

        # ── Angles ────────────────────────────────────────────────────────────
        f.write("\nAngles\n\n")
        for angle_id, (tid, ai, aj, ak) in enumerate(angles, 1):
            f.write(f"     {angle_id}   {tid}      {ai}      {aj}      {ak}\n")

        # ── Dihedrals ─────────────────────────────────────────────────────────
        f.write("\nDihedrals\n\n")
        for dih_id, (tid, ai, aj, ak, al) in enumerate(dihedrals, 1):
            f.write(f"     {dih_id}   {tid}      {ai}      {aj}      {ak}      {al}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MONOMER TOPOLOGY VALIDATION GATE
# ─────────────────────────────────────────────────────────────────────────────
#
# build_polymer_system() builds bonds purely from the LLM-authored intra_bonds
# plus the head->tail inter-monomer link, then enumerates angles/dihedrals from
# that graph. If the LLM mis-wires the monomer (most commonly: a pendant group
# threaded INTO the backbone, so a comb polymer is built as a linear chain),
# the constructor faithfully builds the wrong molecule -- and the OPLS lookups
# silently substitute element-default parameters instead of failing. The bad
# topology then propagates into a full multi-chain build and hours of MD.
#
# This gate runs BEFORE the build and rejects chemically invalid monomers.
# Fatal checks (force-field grounded, no chemistry guessing required):
#   1. every bond (intra + inter-monomer) must have an EXACT OPLS parameter;
#   2. each atom's coordination must match its OPLS type's valence;
#   3. the monomer graph must be connected with distinct, heavy head/tail.
# Missing angle/dihedral parameters are reported as warnings only (the FF may
# legitimately fall back for those).
#
# LIMITATION: for an all-carbon backbone (e.g. poly(4-methyl-1-pentene)) a
# linearized monomer can still have every bond present in OPLS and every carbon
# at valence 4, so it CANNOT be distinguished from a valid linear isomer by
# topology alone. Catching that case requires a connectivity reference
# (SMILES -> RDKit); see the SMILES-based builder route.

# Expected explicit-bond coordination per OPLS bond_type, in the OPLS-AA
# single-bond convention (double/triple bonds are written as single bonds).
# Ranges allow context-variable atoms (e.g. nitrile N = 1 vs imine/azo N = 2).
_TERMINAL_BT = {'HC', 'HA', 'HO', 'HS', 'H', 'F', 'Cl', 'Br', 'I'}
_EXPECTED_COORD = {
    'CT': (4, 4), 'CA': (3, 3), 'CM': (3, 3), 'C': (3, 3), 'CZ': (2, 2),
    'N': (3, 3), 'NT': (3, 3), 'NZ': (1, 2),
    'O': (1, 1), 'OH': (2, 2), 'OS': (2, 2), 'OY': (1, 2),
    'S': (2, 2), 'SH': (2, 2), 'SY2': (4, 4),
}


def _expected_coord(bond_type: str):
    if bond_type in _TERMINAL_BT:
        return (1, 1)
    return _EXPECTED_COORD.get(bond_type)


# ─────────────────────────────────────────────────────────────────────────────
# CRU MOLECULAR-FORMULA CHECK
# ─────────────────────────────────────────────────────────────────────────────
# Charge neutrality and valence checks both pass for a chemically VALID molecule
# that is simply the WRONG molecule. Polyisobutylene -CH2-C(CH3)2- (C4H8) losing
# one of its two methyl branches becomes polypropylene -CH2-CH(CH3)- (C3H6):
# every atom has correct valence, the unit is neutral, and nothing downstream
# objects. The only way to catch that class of error is to compare the built
# repeat unit against an independently obtained molecular formula.

_ELEMENT_MASSES = [
    (1.008, "H"),   (12.011, "C"),  (14.007, "N"),  (15.999, "O"),
    (18.998, "F"),  (22.990, "Na"), (24.305, "Mg"), (28.086, "Si"),
    (30.974, "P"),  (32.060, "S"),  (35.453, "Cl"), (39.098, "K"),
    (79.904, "Br"), (126.904, "I"),
]

# Hill order: C first, then H, then everything else alphabetically.
def _hill_sort_key(sym: str):
    return (0, "") if sym == "C" else (1, "") if sym == "H" else (2, sym)


def element_of_opls_id(opls_id: str, opls_atoms: dict) -> str | None:
    """Element symbol for an OPLS atom key, resolved from its tabulated mass."""
    entry = opls_atoms.get(opls_id)
    if not entry:
        return None
    mass = float(entry.get("mass", 0.0))
    best, best_d = None, float("inf")
    for m, sym in _ELEMENT_MASSES:
        d = abs(mass - m)
        if d < best_d:
            best, best_d = sym, d
    return best if best_d <= 0.35 else None


def cru_formula(poly_def: dict, opls_atoms: dict) -> tuple[dict, str, float]:
    """
    Element counts, Hill-notation formula and mass of the built repeat unit.

    Returns
    -------
    (counts, formula_string, mass_amu)   e.g. ({"C":4,"H":8}, "C4H8", 56.106)
    """
    counts: dict[str, int] = {}
    mass = 0.0
    for atom in poly_def.get("monomer_atoms", []):
        oid = atom.get("opls_id")
        sym = element_of_opls_id(oid, opls_atoms)
        if sym is None:
            continue
        counts[sym] = counts.get(sym, 0) + 1
        mass += float(opls_atoms[oid].get("mass", 0.0))

    parts = []
    for sym in sorted(counts, key=_hill_sort_key):
        n = counts[sym]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return counts, "".join(parts), round(mass, 3)


def parse_formula(formula: str) -> dict:
    """'C4H8' / 'C 4 H 8' / 'C4H8O2' -> {'C':4,'H':8,...}. Unparsable -> {}."""
    if not formula:
        return {}
    text = re.sub(r"[\s\-_()\[\]]", "", str(formula))
    counts: dict[str, int] = {}
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", text):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


def compare_cru_formula(poly_def: dict, opls_atoms: dict,
                        expected_formula: str) -> dict:
    """
    Compare the built CRU against an expected molecular formula.

    Returns
    -------
    dict with keys:
        ok         : bool  (True also when the expectation is unusable)
        built      : Hill formula of the built CRU
        expected   : normalised expected formula
        mass       : built CRU mass (amu)
        diff       : {element: built - expected} for mismatching elements
        checked    : whether a real comparison happened
        message    : human-readable summary (empty when ok)
    """
    built_counts, built_str, mass = cru_formula(poly_def, opls_atoms)
    exp_counts = parse_formula(expected_formula)

    out = {"ok": True, "built": built_str, "expected": expected_formula,
           "mass": mass, "diff": {}, "checked": False, "message": ""}
    if not exp_counts or not built_counts:
        return out                      # nothing to compare against

    out["checked"] = True
    diff = {}
    for sym in set(built_counts) | set(exp_counts):
        d = built_counts.get(sym, 0) - exp_counts.get(sym, 0)
        if d != 0:
            diff[sym] = d
    if not diff:
        return out

    out["ok"] = False
    out["diff"] = diff
    detail = ", ".join(
        f"{sym}: built {built_counts.get(sym,0)} vs expected {exp_counts.get(sym,0)}"
        for sym in sorted(diff, key=_hill_sort_key))
    out["message"] = (
        f"Repeat unit is {built_str} but should be "
        f"{''.join(s if exp_counts[s]==1 else s+str(exp_counts[s]) for s in sorted(exp_counts, key=_hill_sort_key))}"
        f" ({detail})")
    return out


def validate_monomer_topology(poly_def: dict, opls: dict) -> tuple[bool, list, list]:
    """Sanity-check an LLM-authored monomer before a full build.

    Returns (ok, errors, warnings). See the section header for what is and is
    not caught. `errors` being empty means the monomer passed the fatal checks.
    """
    errors: list[str] = []
    warns:  list[str] = []

    atoms = poly_def.get('monomer_atoms', [])
    bonds = [tuple(b) for b in poly_def.get('intra_bonds', [])]
    head  = poly_def.get('head_atom')
    tail  = poly_def.get('tail_atom')

    # local_id -> OPLS bond_type
    lid2bt: dict = {}
    for a in atoms:
        oid = a.get('opls_id')
        lid = a.get('local_id')
        if oid not in opls['atom']:
            errors.append(f"atom local_id={lid}: opls_id '{oid}' not found in atom.txt")
            continue
        lid2bt[lid] = opls['atom'][oid]['bond_type']
    lids = set(lid2bt)

    # adjacency from intra-monomer bonds
    adj = {l: set() for l in lids}
    for (i, j) in bonds:
        if i not in lids or j not in lids:
            errors.append(f"intra_bond ({i},{j}) references an undefined atom")
            continue
        adj[i].add(j); adj[j].add(i)

    # head / tail sanity
    if head == tail:
        errors.append(f"head_atom == tail_atom ({head}); backbone has no direction")
    for nm, x in (('head', head), ('tail', tail)):
        if x not in lids:
            errors.append(f"{nm}_atom={x} is not a defined atom")
        elif _element_of(lid2bt[x]) == 'H':
            errors.append(f"{nm}_atom={x} is hydrogen; backbone connector must be a heavy atom")

    # connectivity: monomer must be a single connected component
    if lids:
        seen, stack = set(), [next(iter(lids))]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj[n] - seen)
        if seen != lids:
            errors.append(f"monomer graph is disconnected; unreachable atoms: {sorted(lids - seen)}")

    # per-atom coordination (periodic chain: head links to previous tail, tail to next head)
    for l in sorted(lids):
        deg = len(adj[l]) + (1 if l == head else 0) + (1 if l == tail else 0)
        bt = lid2bt[l]
        rng = _expected_coord(bt)
        if rng is None:
            warns.append(f"atom {l} ({bt}): no reference valence; coordination not checked")
            continue
        lo, hi = rng
        if not (lo <= deg <= hi):
            want = f"{lo}" if lo == hi else f"{lo}-{hi}"
            errors.append(
                f"atom local_id={l} type={bt}: has {deg} bond(s), expected {want} "
                f"(likely a linearized or mis-wired side group)")

    # every bond must have an EXACT OPLS parameter (intra + inter-monomer tail->head)
    bond_keys = list(bonds)
    if head in lid2bt and tail in lid2bt:
        bond_keys.append((tail, head))
    for (i, j) in bond_keys:
        if i not in lid2bt or j not in lid2bt:
            continue
        bi, bj = lid2bt[i], lid2bt[j]
        if (bi, bj) not in opls['bond']:
            tag = " (inter-monomer)" if (i, j) == (tail, head) else ""
            errors.append(
                f"bond {bi}-{bj} (atoms {i}-{j}){tag}: no OPLS bond parameter; "
                f"non-physical connection -- monomer likely mis-wired")

    # junction-aware angle parameter coverage (warn-only: FF may default)
    if lids and head in lids and tail in lids:
        N = max(lids)
        def gid(m, l): return m * N + l
        edges = []
        for m in range(3):
            edges += [(gid(m, i), gid(m, j)) for (i, j) in bonds]
            if m < 2:
                edges.append((gid(m, tail), gid(m + 1, head)))
        ja = {}
        for (i, j) in edges:
            ja.setdefault(i, set()).add(j)
            ja.setdefault(j, set()).add(i)
        bt_of = {gid(m, l): lid2bt[l] for m in range(3) for l in lids}
        missing_ang, seen_ang = set(), set()
        for c in ja:
            nb = sorted(ja[c])
            for x in range(len(nb)):
                for y in range(x + 1, len(nb)):
                    i, k = nb[x], nb[y]
                    key = (min(i, k), c, max(i, k))
                    if key in seen_ang:
                        continue
                    seen_ang.add(key)
                    trip = (bt_of[i], bt_of[c], bt_of[k])
                    if trip not in opls['angle']:
                        missing_ang.add(trip)
        for trip in sorted(missing_ang):
            warns.append(f"angle {trip[0]}-{trip[1]}-{trip[2]}: no exact OPLS param (FF default used)")

    return (len(errors) == 0, errors, warns)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_and_write(poly_def: dict | str,
                    opls_dir: str,
                    output_path: str,
                    n_chains: int = 20,
                    n_monomers: int = 10,
                    density: float = 0.9,
                    seed: int | None = None) -> dict:
    """
    High-level entry point.

    Parameters
    ----------
    poly_def   : polymer topology dict (or JSON string)
    opls_dir   : path to directory containing atom.txt / bond.txt / angle.txt / dihedral.txt
    output_path: output .data file path
    n_chains   : number of polymer chains (default 20)
    n_monomers : monomers per chain        (default 10)
    density    : target density g/cm³      (default 0.9)
    seed       : random seed for reproducibility

    Returns
    -------
    system dict (can be inspected for debug info)
    """
    if isinstance(poly_def, str):
        poly_def = json.loads(poly_def)

    opls   = load_opls(opls_dir)

    # ── Validate the monomer topology BEFORE committing to a full build ──────
    ok, errs, warns = validate_monomer_topology(poly_def, opls)
    for w in warns:
        warnings.warn(f"[topology] {w}")
    if not ok:
        name = poly_def.get('polymer_name', 'unknown')
        msg = (f"Invalid monomer topology for '{name}' -- refusing to build.\n  - "
               + "\n  - ".join(errs)
               + "\n(The monomer graph from the formatter is chemically inconsistent; "
                 "regenerate poly_def rather than simulating a mis-wired structure.)")
        raise ValueError(msg)

    system = build_polymer_system(poly_def, opls,
                                   n_chains=n_chains, n_monomers=n_monomers,
                                   density=density, seed=seed)
    write_lammps_datafile(system, opls, output_path)

    n_atoms     = len(system['atoms'])
    n_bonds     = len(system['bonds'])
    n_angles    = len(system['angles'])
    n_dihedrals = len(system['dihedrals'])
    box_L       = system['box_L']

    print(f"\n✅  Data file written: {output_path}")
    print(f"   Polymer   : {poly_def.get('polymer_name', 'unknown')}")
    print(f"   Chains    : {n_chains} × {n_monomers} monomers")
    print(f"   Monomer M : {system['monomer_mass']:.3f} g/mol")
    print(f"   Box size  : {box_L:.4f} Å  (density {density} g/cm³)")
    print(f"   Atoms     : {n_atoms}  |  Bonds: {n_bonds}  |  "
          f"Angles: {n_angles}  |  Dihedrals: {n_dihedrals}")
    print(f"   Atom types: {len(system['seen_types'])}  |  "
          f"Bond types: {len(system['bond_type_keys'])}  |  "
          f"Angle types: {len(system['angle_type_keys'])}  |  "
          f"Dihedral types: {len(system['dih_type_keys'])}")

    return system


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python aa_constructor.py <poly_def.json> <opls_dir> <output.data>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        pdef = json.load(f)
    build_and_write(pdef, sys.argv[2], sys.argv[3])
