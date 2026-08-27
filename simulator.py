"""
simulator.py
============
Simulation tool for CGMas.

Public functions:
  build_in_file(...)      – generate LAMMPS input script from AAEQ/CGEQ template
  build_tg_in_file(...)   – generate LAMMPS input script for CG Tg annealing
  build_aa_tg_in_file(...)– generate LAMMPS input script for AA Tg annealing
  run_lammps(...)         – execute LAMMPS via Python bindings (MPI)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_T_INIT   = 300    # K  operating temperature
DEFAULT_T_MAX    = 500    # K  annealing peak temperature
DEFAULT_N_ANNEAL = 1      # number of annealing cycles
DEFAULT_EQUIL_PS = 1000   # ps  final NPT equilibration time
DEFAULT_ANNEAL_SPAN = 200 # K   Tfin - Tini fallback when only Tini is given

STEPS_PER_PS     = 1000   # timestep = 1 fs → 1000 steps = 1 ps


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE HELPERS  (shared behaviour with cg_constructor.build_cg_in_file)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_temps(t_init, t_max, *, label: str = "AA"):
    """Validate/clean a (Tini, Tfin) pair coming from a user query.

    * accepts str/float ("350", "350.0 K") and returns ints
    * Tfin < Tini is physically wrong for the annealing ramp -> Tfin is lifted
      to Tini + DEFAULT_ANNEAL_SPAN and a notice is printed
    * non-positive / unparsable values fall back to the module defaults
    """
    def _num(v, fallback):
        if v is None:
            return fallback
        try:
            f = float(str(v).strip().rstrip("Kk").strip())
        except (TypeError, ValueError):
            return fallback
        return f if f > 0 else fallback

    ti = int(round(_num(t_init, DEFAULT_T_INIT)))
    tf = int(round(_num(t_max,  DEFAULT_T_MAX)))
    if tf < ti:
        tf_new = ti + DEFAULT_ANNEAL_SPAN
        print(f"[{label}] Tfin ({tf} K) < Tini ({ti} K); "
              f"raising Tfin to {tf_new} K for the annealing ramp.")
        tf = tf_new
    return ti, tf


def _ensure_temp_vars(patched: list, t_init: int, t_max: int,
                      seen_tini: bool, seen_tfin: bool) -> list:
    """Insert `variable Tini/Tfin equal ...` if the template had no such lines.

    Without this a template that hard-codes temperatures (or names the
    variables differently) would silently ignore the requested temperature.
    The definitions are placed just before the first line that USES ${Tini} /
    ${Tfin} (e.g. `velocity all create ${Tini} ...`), so they are always in
    scope.
    """
    if seen_tini and seen_tfin:
        return patched

    inject = []
    if not seen_tini:
        inject.append(f"variable Tini   equal {t_init}\n")
    if not seen_tfin:
        inject.append(f"variable Tfin   equal {t_max}\n")

    use_idx = next((i for i, ln in enumerate(patched)
                    if "${Tini}" in ln or "${Tfin}" in ln), None)
    if use_idx is None:
        # no usage either -> put them after the last leading comment/blank line
        use_idx = next((i for i, ln in enumerate(patched)
                        if ln.strip() and not ln.lstrip().startswith("#")), 0)
    print(f"[Simulator] Template had no explicit "
          f"{'Tini ' if not seen_tini else ''}{'Tfin' if not seen_tfin else ''}"
          f" variable; injecting it so the requested temperature is applied.")
    return patched[:use_idx] + inject + patched[use_idx:]


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT DETECTION FROM OPLS BOND-TYPE NAME
# ─────────────────────────────────────────────────────────────────────────────

# Map first character(s) of bond_type → element symbol
_BT_PREFIX_MAP = [
    ("Br", "Br"),
    ("Cl", "Cl"),
    ("Na", "Na"),
    ("Ca", "Ca"),
    ("Sr", "Sr"),
    ("Cs", "Cs"),
    ("Nd", "Nd"),
    ("H",  "H"),
    ("O",  "O"),
    ("N",  "N"),
    ("S",  "S"),
    ("F",  "F"),
    ("P",  "P"),
    ("I",  "I"),
    ("C",  "C"),
]


def _bond_type_to_element(bond_type: str) -> str:
    """Convert OPLS bond_type string (e.g. 'CT', 'HC', 'OS') to element symbol."""
    bt = bond_type.strip()
    for prefix, element in _BT_PREFIX_MAP:
        if bt.startswith(prefix):
            return element
    return bt[0].upper() if bt else "X"


# Atomic masses (amu) for the elements that show up in OPLS-AA systems.
# Needed because LAMMPS `write_data` drops the "# CT" comments that the
# constructor writes, so an equilibrated AAEQfin.data has bare "1 12.011"
# lines and the element symbol has to be recovered from the mass itself.
_MASS_TO_ELEMENT = [
    (1.008, "H"),   (4.003, "He"),  (6.941, "Li"),  (9.012, "Be"),
    (10.811, "B"),  (12.011, "C"),  (14.007, "N"),  (15.999, "O"),
    (18.998, "F"),  (20.180, "Ne"), (22.990, "Na"), (24.305, "Mg"),
    (26.982, "Al"), (28.086, "Si"), (30.974, "P"),  (32.060, "S"),
    (35.453, "Cl"), (39.098, "K"),  (39.948, "Ar"), (40.078, "Ca"),
    (55.845, "Fe"), (63.546, "Cu"), (65.380, "Zn"), (79.904, "Br"),
    (126.904, "I"),
]

_MASS_TOL = 0.35   # amu — tight enough to keep K (39.098) and Ar (39.948) apart


def _mass_to_element(mass: float) -> str | None:
    """Nearest element symbol for an atomic mass, or None if nothing is close."""
    best, best_diff = None, float("inf")
    for m, sym in _MASS_TO_ELEMENT:
        d = abs(mass - m)
        if d < best_diff:
            best, best_diff = sym, d
    return best if best_diff <= _MASS_TOL else None


def parse_masses_section(datafile_path: str) -> list[str]:
    """
    Read the Masses section of a LAMMPS .data file and return an ordered list
    of element symbols corresponding to atom types 1, 2, 3, …

    Two layouts are supported:

      1. constructor output (has the bond_type comment)
             1  12.011000 # CT  (opls_064)
      2. LAMMPS `write_data` output (comments stripped)
             1 12.011

    Layout 2 is what AAEQfin.data looks like after an equilibration run, so the
    element is recovered from the mass when no comment is present. Types are
    returned in atom-type order (1, 2, 3, …) regardless of the order of the
    lines in the file.
    """
    by_type: dict[int, str] = {}
    in_masses = False

    with open(datafile_path) as f:
        for line in f:
            stripped = line.strip()

            if stripped.lower().startswith("masses"):
                in_masses = True
                continue

            if in_masses:
                if stripped == "" or (stripped and stripped[0].isalpha() and
                                      not stripped.startswith("#")):
                    # Empty line or new section header → done
                    if stripped != "":
                        break
                    continue

                if stripped.startswith("#"):
                    continue

                # <type_id>  <mass>  [# <bond_type>  (<opls_id>)]
                m = re.match(r'^\s*(\d+)\s+([\d.eEdD+-]+)\s*(?:#\s*(\S+))?', stripped)
                if not m:
                    continue
                type_id, mass_str, bond_type = m.group(1), m.group(2), m.group(3)

                if bond_type:
                    sym = _bond_type_to_element(bond_type)
                else:
                    try:
                        sym = _mass_to_element(float(mass_str.replace("D", "E")))
                    except ValueError:
                        sym = None
                    if sym is None:
                        # Unknown mass (e.g. a CG bead): fall back to a stable
                        # per-type label so dump_modify still gets one token per
                        # atom type and LAMMPS does not error out.
                        sym = f"T{type_id}"
                        print(f"[Simulator] WARNING: Masses type {type_id} "
                              f"(mass {mass_str}) matches no element; "
                              f"using placeholder '{sym}' in dump_modify.")
                by_type[int(type_id)] = sym

    return [by_type[k] for k in sorted(by_type)]


# ─────────────────────────────────────────────────────────────────────────────
# IN. FILE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

_ANNEAL_BLOCK_TEMPLATE = """\
# ----- Temp increase  [cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tini}} ${{Tfin}} 1000.0
dump           1 all custom 10000 heating_{cyc}.lammpstrj id element xu yu zu vx vy vz
dump_modify    1 element {elements}
run            100000

unfix     1
undump    1

# ----- NVT  [holding, cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tfin}} ${{Tfin}} 1000.0
dump           1 all custom 10000 holding_{cyc}.lammpstrj id element xu yu zu vx vy vz
dump_modify    1 element {elements}
run            100000

unfix     1
undump    1

# ----- Temp decrease  [cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tfin}} ${{Tini}} 1000.0
dump           1 all custom 10000 cooling_{cyc}.lammpstrj id element xu yu zu vx vy vz
dump_modify    1 element {elements}
run            100000

unfix     1
undump    1

# ----- NVT  [relaxing, cycle {cyc}]
reset_timestep 0
fix            1 all nvt temp ${{Tini}} ${{Tini}} 1000.0
dump           1 all custom 10000 relaxing_{cyc}.lammpstrj id element xu yu zu vx vy vz
dump_modify    1 element {elements}
run            100000

unfix     1
undump    1
"""


def build_in_file(
    template_path: str,
    datafile_path: str,
    output_path: str,
    t_init:   int = DEFAULT_T_INIT,
    t_max:    int = DEFAULT_T_MAX,
    n_anneal: int = DEFAULT_N_ANNEAL,
    equil_ps: int = DEFAULT_EQUIL_PS,
) -> str:
    """
    Generate a LAMMPS input script from the AAEQ template.

    Parameters
    ----------
    template_path : path to in.AAEQ template
    datafile_path : path to the .data file produced by constructor.py
    output_path   : where to write the output in. file
    t_init        : operating / initial temperature (K)
    t_max         : annealing peak temperature (K)
    n_anneal      : number of annealing cycles
    equil_ps      : final NPT equilibration duration (ps)

    Returns
    -------
    output_path
    """
    # 0. Normalise the requested temperatures (query values may be str/float,
    #    and Tfin must stay above Tini for the annealing ramp).
    t_init, t_max = normalize_temps(t_init, t_max, label="AA")

    # 1. Derive element string from .data file
    elements_list = parse_masses_section(datafile_path)
    if not elements_list:
        raise ValueError(
            f"Could not parse Masses section from {datafile_path}. "
            "Check that the file exists and has a Masses section with # comments."
        )
    elements_str = " ".join(elements_list)

    # 2. Derive data file basename (LAMMPS reads relative to run dir)
    data_basename = os.path.basename(datafile_path)

    # 3. Compute equil steps
    equil_steps = equil_ps * STEPS_PER_PS

    # 4. Read and patch the template
    with open(template_path) as f:
        lines = f.readlines()

    # Replace variable lines and read_data
    patched: list[str] = []
    seen_tini = seen_tfin = False
    for line in lines:
        if re.match(r'\s*variable\s+Tini\s+equal', line, re.IGNORECASE):
            patched.append(f"variable Tini   equal {t_init}\n")
            seen_tini = True
        elif re.match(r'\s*variable\s+Tfin\s+equal', line, re.IGNORECASE):
            patched.append(f"variable Tfin   equal {t_max}\n")
            seen_tfin = True
        elif re.match(r'\s*variable\s+equil\s+equal', line):
            patched.append(f"variable equil  equal {equil_steps}\n")
        elif re.match(r'\s*read_data', line):
            patched.append(f"read_data       {data_basename}\n")
        elif re.match(r'\s*dump_modify', line):
            # Replace element list after "element"
            patched.append(
                re.sub(r'(dump_modify\s+\S+\s+element\s+).*',
                       rf'\g<1>{elements_str}', line)
            )
        else:
            patched.append(line)

    patched = _ensure_temp_vars(patched, t_init, t_max, seen_tini, seen_tfin)
    template_text = "".join(patched)

    print(f"[Simulator] in.AAEQ  ->  Tini = {t_init} K | Tfin = {t_max} K "
          f"| {n_anneal} annealing cycle(s) | {equil_ps} ps EQ")

    # 5. Handle multiple annealing cycles
    # The template contains exactly 1 cycle (NVT init → heat → hold → cool →
    # relax → NPT). For n_anneal > 1 we replicate the annealing block.
    if n_anneal > 1:
        # Find the block between "# ----- Temp increase" and "# ----- NPT"
        m = re.search(
            r'(# ----- Temp increase.*?)(# ----- NPT)',
            template_text, re.DOTALL
        )
        if m:
            anneal_blocks = ""
            for cyc in range(1, n_anneal + 1):
                anneal_blocks += _ANNEAL_BLOCK_TEMPLATE.format(
                    cyc=cyc, elements=elements_str
                )
            template_text = (
                template_text[:m.start(1)]
                + anneal_blocks
                + template_text[m.start(2):]
            )

    # 6. Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template_text)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Tg ANNEALING INPUT FILE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TG_MARGIN_K = 120   # K  ← Tg ± 120 K for Tini / Tfin
_TINI_FLOOR         = 80    # K  물리적 하한선


def calc_tg_temps(
    ref_tg_k: float,
    margin_k: float = DEFAULT_TG_MARGIN_K,
) -> tuple[int, int]:
    """
    참고 Tg를 기반으로 아닐링 시뮬레이션의 Tini / Tfin을 계산합니다.

        Tini = max(TINI_FLOOR, round(ref_tg_k - margin_k))
        Tfin = round(ref_tg_k + margin_k)

    Parameters
    ----------
    ref_tg_k : LLM 참고 Tg (K)
    margin_k : Tg 앞뒤 온도 범위 (K, 기본 120)

    Returns
    -------
    (t_ini, t_fin) : int, int
    """
    t_ini = max(_TINI_FLOOR, round(ref_tg_k - margin_k))
    t_fin = round(ref_tg_k + margin_k)
    return int(t_ini), int(t_fin)


def _is_multi_type_key(k) -> bool:
    """
    True if k is a tuple/list key OR a string encoding of one
    (e.g. '[1, 1]', '(1, 2)').

    Needed because potential_results is serialised in LangGraph state
    with string keys (e.g. '[1,1]') but _build_ff_coeff_lines also
    receives de-serialised results with real tuple keys.
    """
    if isinstance(k, (tuple, list)):
        return True
    if isinstance(k, str):
        import ast
        try:
            v = ast.literal_eval(k)
            return isinstance(v, (list, tuple))
        except Exception:
            pass
    return False


def _build_ff_style_lines() -> list[str]:
    """
    LAMMPS force-field style 선언만 반환합니다 (read_data 앞에 삽입).
    coefficients 는 포함하지 않습니다.
    """
    return [
        "bond_style      harmonic\n",
        "angle_style     harmonic\n",
        "pair_style      lj/cut 15\n",
    ]


def _build_ff_coeff_lines(potential_results: dict) -> list[str]:
    """
    potential.py 결과 dict에서 LAMMPS force-field 계수 라인을 생성합니다.
    (style 선언 포함 — read_data 뒤에 오는 경우 _build_ff_style_lines 와 함께 쓰지 말 것)

    단일 타입(단순 dict):
        {"bond":  {"r0": float, "k": float},
         "angle": {"theta0": float, "k": float},
         "lj":    {"epsilon": float, "sigma": float}}

    다중 타입(typed dict):
        {"bond":  {(1,2): {"r0": ..., "k": ...}, ...},
         "angle": {(1,1,2): {"theta0": ..., "k": ...}, ...},
         "lj":    {(1,1): {"epsilon": ..., "sigma": ...}, ...}}
    """
    lines: list[str] = []

    bond_data  = potential_results.get("bond",  {})
    angle_data = potential_results.get("angle", {})
    lj_data    = potential_results.get("lj",    {})

    # ── bond_style / bond_coeff ─────────────────────────────────────────────
    lines.append("bond_style      harmonic\n")
    if bond_data:
        first_key = next(iter(bond_data))
        if _is_multi_type_key(first_key):
            for idx, (btype, p) in enumerate(bond_data.items(), start=1):
                lines.append(
                    f"bond_coeff      {idx} {p['k']:.6f} {p['r0']:.6f}"
                    f"  # {btype}\n"
                )
        else:
            lines.append(
                f"bond_coeff      1 {bond_data['k']:.6f} {bond_data['r0']:.6f}\n"
            )

    # ── angle_style / angle_coeff ───────────────────────────────────────────
    lines.append("\nangle_style     harmonic\n")
    if angle_data:
        first_key = next(iter(angle_data))
        if _is_multi_type_key(first_key):
            for idx, (atype, p) in enumerate(angle_data.items(), start=1):
                lines.append(
                    f"angle_coeff     {idx} {p['k']:.6f} {p['theta0']:.6f}"
                    f"  # {atype}\n"
                )
        else:
            lines.append(
                f"angle_coeff     1 {angle_data['k']:.6f} {angle_data['theta0']:.6f}\n"
            )

    # ── pair_style / pair_coeff ─────────────────────────────────────────────
    lines.append("\npair_style      lj/cut 15\n")
    if lj_data:
        first_key = next(iter(lj_data))
        if _is_multi_type_key(first_key):
            for lj_type, p in lj_data.items():
                i, j = (lj_type if isinstance(lj_type, (tuple, list))
                         else __import__("ast").literal_eval(lj_type))
                lines.append(
                    f"pair_coeff      {i} {j} {p['epsilon']:.6f} {p['sigma']:.6f}"
                    f"  # {lj_type}\n"
                )
        else:
            lines.append(
                f"pair_coeff      1 1 {lj_data['epsilon']:.6f} {lj_data['sigma']:.6f}\n"
            )

    return lines


def _build_ff_coeff_only_lines(potential_results: dict) -> list[str]:
    """
    coeff 라인만 반환합니다 (style 선언 제외).
    read_data 뒤, style 선언이 이미 앞에 있을 때 사용합니다.
    """
    lines: list[str] = []

    bond_data  = potential_results.get("bond",  {})
    angle_data = potential_results.get("angle", {})
    lj_data    = potential_results.get("lj",    {})

    if bond_data:
        first_key = next(iter(bond_data))
        if _is_multi_type_key(first_key):
            for idx, (btype, p) in enumerate(bond_data.items(), start=1):
                lines.append(
                    f"bond_coeff      {idx} {p['k']:.6f} {p['r0']:.6f}"
                    f"  # {btype}\n"
                )
        else:
            lines.append(
                f"bond_coeff      1 {bond_data['k']:.6f} {bond_data['r0']:.6f}\n"
            )

    if angle_data:
        first_key = next(iter(angle_data))
        if _is_multi_type_key(first_key):
            for idx, (atype, p) in enumerate(angle_data.items(), start=1):
                lines.append(
                    f"angle_coeff     {idx} {p['k']:.6f} {p['theta0']:.6f}"
                    f"  # {atype}\n"
                )
        else:
            lines.append(
                f"angle_coeff     1 {angle_data['k']:.6f} {angle_data['theta0']:.6f}\n"
            )

    if lj_data:
        first_key = next(iter(lj_data))
        if _is_multi_type_key(first_key):
            for lj_type, p in lj_data.items():
                i, j = (lj_type if isinstance(lj_type, (tuple, list))
                         else __import__("ast").literal_eval(lj_type))
                lines.append(
                    f"pair_coeff      {i} {j} {p['epsilon']:.6f} {p['sigma']:.6f}"
                    f"  # {lj_type}\n"
                )
        else:
            lines.append(
                f"pair_coeff      1 1 {lj_data['epsilon']:.6f} {lj_data['sigma']:.6f}\n"
            )

    return lines


def build_tg_in_file(
    template_path: str,
    output_path: str,
    potential_results: dict,
    t_ini: int,
    t_fin: int,
    datafile: str = "",
) -> str:
    """
    in.CGTg 템플릿을 그대로 복사하되,
    bond_coeff / angle_coeff / pair_coeff 숫자, Tini/Tfin, read_data 파일명만 교체.
    """
    def _vals(d):
        if not d:
            return []
        first = next(iter(d))
        if isinstance(first, tuple):
            out = []
            for p in d.values():
                v1 = p.get("k", p.get("epsilon"))
                v2 = p.get("r0", p.get("sigma", p.get("theta0")))
                out.append((v1, v2))
            return out
        v1 = d.get("k", d.get("epsilon"))
        v2 = d.get("r0", d.get("sigma", d.get("theta0")))
        return [(v1, v2)]

    b_iter  = iter(_vals(potential_results.get("bond",  {})))
    a_iter  = iter(_vals(potential_results.get("angle", {})))
    lj_iter = iter(_vals(potential_results.get("lj",    {})))

    with open(template_path) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        low = line.strip().lower()

        if low.startswith("bond_coeff"):
            m = re.match(r"^(\s*bond_coeff\s+(?:\d+\s+)+)", line, re.IGNORECASE)
            try:
                v1, v2 = next(b_iter)
                prefix = m.group(1) if m else "bond_coeff 1 "
                out_lines.append(prefix + "%.6f %.6f\n" % (v1, v2))
            except StopIteration:
                out_lines.append(line)

        elif low.startswith("angle_coeff"):
            m = re.match(r"^(\s*angle_coeff\s+(?:\d+\s+)+)", line, re.IGNORECASE)
            try:
                v1, v2 = next(a_iter)
                prefix = m.group(1) if m else "angle_coeff 1 "
                out_lines.append(prefix + "%.6f %.6f\n" % (v1, v2))
            except StopIteration:
                out_lines.append(line)

        elif low.startswith("pair_coeff"):
            m = re.match(r"^(\s*pair_coeff\s+(?:\d+\s+)+)", line, re.IGNORECASE)
            try:
                v1, v2 = next(lj_iter)
                prefix = m.group(1) if m else "pair_coeff 1 1 "
                out_lines.append(prefix + "%.6f %.6f\n" % (v1, v2))
            except StopIteration:
                out_lines.append(line)

        elif re.match(r"\s*variable\s+Tini\s+equal", line, re.IGNORECASE):
            out_lines.append(re.sub(r"(equal\s+)\S+", r"\g<1>" + str(t_ini), line))

        elif re.match(r"\s*variable\s+Tfin\s+equal", line, re.IGNORECASE):
            out_lines.append(re.sub(r"(equal\s+)\S+", r"\g<1>" + str(t_fin), line))

        elif datafile and re.match(r"\s*read_data", line, re.IGNORECASE):
            out_lines.append(re.sub(r"(read_data\s+)\S+",
                                    r"\g<1>" + os.path.basename(datafile), line))
        else:
            out_lines.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(out_lines)

    print("[Simulator] Tg input file written: " + output_path)
    print("            Tini = %d K  |  Tfin = %d K" % (t_ini, t_fin))
    return output_path



def build_aa_tg_in_file(
    template_path: str,
    output_path: str,
    datafile: str,
    t_ini: int,
    t_fin: int,
    elements_from: str = None,
) -> str:
    """
    프로젝트에 미리 들어있는 in.AATg 템플릿을 **수정**해서 AA Tg 아닐링
    입력 파일을 만듭니다. 스크립트를 새로 생성하지 않습니다 -- 템플릿의
    run 길이, fix/dump 설정, thermo 설정 등은 그대로 보존되고 아래 네 종류의
    줄만 교체됩니다. (in.AAEQ 를 다루는 build_in_file 과 완전히 동일한 방식)

      * variable Tini / Tfin      -> Tg ± margin 온도
      * read_data                 -> datafile 의 basename (보통 AAEQfin.data)
      * dump_modify ... element X -> .data 의 Masses 섹션에서 뽑은 원소 목록
                                     (예: "element C"  ->  "element C H O")

    CG용 build_tg_in_file 과 달리 force-field 계수(bond/angle/pair_coeff)는
    건드리지 않습니다. AA 시스템의 계수는 AAEQfin.data 안에 이미 있습니다.

    Parameters
    ----------
    template_path : in.AATg 템플릿 경로 (프로젝트 폴더에 있는 파일)
    output_path   : 생성할 in. 파일 경로 (세션 폴더)
    datafile      : 아닐링을 돌릴 .data 파일 (보통 AAEQfin.data)
    t_ini, t_fin  : 아닐링 시작/끝 온도 (K)
    elements_from : element 목록을 뽑아올 .data 파일. 생략하면 datafile 사용.
                    LAMMPS write_data 는 Masses 의 "# CT" 주석을 지우므로,
                    constructor 가 만든 원본 .data 를 넘기면 AA EQ 와 완전히
                    동일한 element 목록이 보장됩니다.

    Returns
    -------
    output_path
    """
    # ── 0. 입력 확인 ─────────────────────────────────────────────────────────
    if not template_path or not os.path.exists(template_path):
        raise FileNotFoundError(
            f"AA Tg template not found: {template_path!r}. "
            "in.AATg must sit next to simulator.py, alongside "
            "in.AAEQ / in.CGEQ / in.CGTg."
        )
    if not datafile or not os.path.exists(datafile):
        raise FileNotFoundError(
            f"AA Tg annealing needs an equilibrated AA data file; got: {datafile!r}. "
            "Expected AAEQfin.data from the AA equilibration run."
        )

    # AA EQ와 똑같이 Masses 섹션에서 element 문자열을 만든다.
    # 우선순위: constructor 원본 .data(주석 포함) -> 아닐링용 .data(주석 없음,
    # 질량으로 원소 추론). 두 파일의 원자 타입 순서는 동일하다.
    src_candidates = [c for c in (elements_from, datafile) if c and os.path.exists(c)]
    elements_list, elements_src = [], None
    for cand in src_candidates:
        try:
            got = parse_masses_section(cand)
        except Exception:
            got = []
        if got:
            elements_list, elements_src = got, cand
            break
    if not elements_list:
        raise ValueError(
            "Could not parse a Masses section from any of: "
            + ", ".join(src_candidates)
            + ". dump_modify element list cannot be built."
        )
    if elements_from and elements_src != elements_from:
        print(f"[Simulator] NOTE: element list taken from "
              f"{os.path.basename(elements_src)} "
              f"(masses inferred; {os.path.basename(elements_from)} unusable).")
    elements_str  = " ".join(elements_list)
    data_basename = os.path.basename(datafile)

    # ── 1. 템플릿 읽고 해당 줄만 교체 ────────────────────────────────────────
    with open(template_path) as f:
        lines = f.readlines()

    out_lines: list[str] = []
    n_tini = n_tfin = n_read = n_dump = 0
    for line in lines:
        if re.match(r"\s*variable\s+Tini\s+equal", line, re.IGNORECASE):
            out_lines.append(f"variable Tini   equal {t_ini}\n")
            n_tini += 1
        elif re.match(r"\s*variable\s+Tfin\s+equal", line, re.IGNORECASE):
            out_lines.append(f"variable Tfin   equal {t_fin}\n")
            n_tfin += 1
        elif re.match(r"\s*read_data", line, re.IGNORECASE):
            out_lines.append(f"read_data       {data_basename}\n")
            n_read += 1
        elif re.match(r"\s*dump_modify", line, re.IGNORECASE):
            # Only dump_modify lines carrying an "element" list are rewritten;
            # e.g. "dump_modify 1 sort id" passes through untouched.
            patched_line, n_sub = re.subn(
                r"(dump_modify\s+\S+\s+element\s+).*",
                rf"\g<1>{elements_str}", line)
            out_lines.append(patched_line)
            n_dump += n_sub
        else:
            out_lines.append(line)

    # 템플릿에 Tini/Tfin 변수 선언이 없으면 첫 사용처 앞에 삽입 (온도 무시 방지)
    out_lines = _ensure_temp_vars(out_lines, t_ini, t_fin,
                                  n_tini > 0, n_tfin > 0)

    # ── 2. 후속 단계가 의존하는 요소 확인 ────────────────────────────────────
    body = "".join(out_lines)
    if n_read == 0:
        print("[Simulator] WARNING: in.AATg has no read_data line; "
              f"LAMMPS will not load {data_basename}.")
    if n_dump == 0:
        print("[Simulator] WARNING: in.AATg has no dump_modify line; "
              "the element list could not be applied.")
    # calc_tg_from_annealing 은 로그에서 이 주석 마커로 가열/냉각 구간을 나눈다.
    missing = [m for m in ("# ----- Temp increase", "# ----- Temp decrease")
               if m not in body]
    if missing:
        print("[Simulator] WARNING: in.AATg is missing the marker(s) "
              + ", ".join(repr(m) for m in missing)
              + " -- Tg fitting splits heating/cooling on these comments, "
                "so the AA Tg calculation will fail without them.")

    # ── 3. 출력 ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(out_lines)

    print("[Simulator] AA Tg input file written: " + output_path)
    print("            template    = %s (patched, not regenerated)"
          % os.path.basename(template_path))
    print("            Tini = %d K  |  Tfin = %d K   (%d/%d lines)"
          % (t_ini, t_fin, n_tini, n_tfin))
    print("            read_data   = %s   (%d line%s)"
          % (data_basename, n_read, "" if n_read == 1 else "s"))
    print("            elements    = %s   (%d dump_modify line%s, from %s)"
          % (elements_str, n_dump, "" if n_dump == 1 else "s",
             os.path.basename(elements_src)))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# LAMMPS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_lammps(
    in_file: str,
    datafile: str,
    output_dir: str,
    n_cores: int = 1,
    label: str = "",
) -> dict:
    """
    Run LAMMPS via CLI subprocess.
    cwd is set to output_dir so dump/trj files land there.

    Parameters
    ----------
    label : optional suffix appended to log/screen filenames.
            e.g. label="aa"  →  lammps_aa.log / lammps_aa.screen
                 label="cg"  →  lammps_cg.log / lammps_cg.screen
            If empty (default), filenames stay as lammps.log / lammps.screen.
    """
    import shutil, subprocess

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    data_dest = out / Path(datafile).name
    if Path(datafile).resolve() != data_dest:
        shutil.copy2(datafile, data_dest)

    in_dest = out / Path(in_file).name
    if Path(in_file).resolve() != in_dest:
        shutil.copy2(in_file, in_dest)

    if not in_dest.exists():
        raise FileNotFoundError(f"Input file not found: {in_dest}")
    if not data_dest.exists():
        raise FileNotFoundError(f"Data file not found: {data_dest}")

    _suffix     = f"_{label}" if label else ""
    log_path    = out / f"lammps{_suffix}.log"
    screen_path = out / f"lammps{_suffix}.screen"

    print(f"[LAMMPS] Run dir    : {out}")
    print(f"[LAMMPS] Input file : {in_dest.name}")
    print(f"[LAMMPS] Data file  : {data_dest.name}")
    print("[LAMMPS] Running simulation (this may take a while)...")
    print("-" * 60, flush=True)

    # Run with cwd=out so relative paths (read_data, dump) resolve correctly.
    # stdout/stderr stream live to the notebook; log is written by -log flag.
    # stdin=DEVNULL prevents LAMMPS from blocking on empty stdin.
    # Use Python bindings with version-check bypass.
    # The mismatch is only a safety check — the actual API is compatible.

    # mpirun으로 실행할 임시 python 스크립트 생성
    import sys as _sys
    _runner_script = (
        "import os, lammps as _lm\n"
        "_lm.__version__ = 0\n"
        f'os.chdir(r"{out}")\n'
        f'lmp = _lm.lammps(cmdargs=["-log", r"{log_path}", "-screen", r"{screen_path}"])\n'
        f'lmp.file(r"{in_dest}")\n'
        "lmp.close()\n"
    )
    _tmp = out / "_mpi_runner.py"
    _tmp.write_text(_runner_script)

    python_bin = str(Path(_sys.executable))

    try:
        result = subprocess.run(
            ["mpirun", "-np", str(n_cores), python_bin, str(_tmp)],
            cwd=str(out),
        )
        returncode = result.returncode
    except Exception as e:
        returncode = 1
        print(f"[LAMMPS] ERROR: {e}")
        raise
    finally:
        _tmp.unlink(missing_ok=True)

    print("-" * 60, flush=True)
    print(f"[LAMMPS] Done. Return code: {returncode}")

    return {
        "log_path":    str(log_path),
        "screen_path": str(screen_path),
        "returncode":  returncode,
    }
