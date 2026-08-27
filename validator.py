"""
validator.py
============
CGMas 시뮬레이션 검증 도구.

LLM을 통해 폴리머 참고값(밀도, Tg)을 가져오고,
AA/CG 시뮬레이션 결과와 비교하여 PASS/FAIL 판정을 출력합니다.

Public functions:
  get_reference_density(polymer_name, llm)     – LLM으로 밀도 참고값 조회
  get_reference_tg(polymer_name, llm)          – LLM으로 Tg 참고값 조회 (K)
  validate_density(ref, aa, cg, ...)           – 밀도를 참고값과 비교 (선택)
  compare_aa_cg_density(aa, cg, ...)           – AA 기준 CG 밀도 오차 (참고값 미사용)
  compare_aa_cg_tg(aa_tg, cg_tg, ...)          – AA 기준 CG Tg 오차 (참고값 미사용)
  calc_tg_from_annealing(log_path, ...)        – 아닐링 로그에서 Tg 계산
  validate_tg(ref, sim_tg, ...)                – Tg 비교 + PASS/FAIL
  run_validation(...)                           – 밀도 검증 원스텝 실행
"""

from __future__ import annotations

import re
import json
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (사용자가 override 가능)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DENSITY_TOL_PCT = 5.0   # ±5 %  이내이면 PASS
DEFAULT_TG_TOL_K        = 15.0  # ±15 K 이내이면 PASS


# ─────────────────────────────────────────────────────────────────────────────
# LLM REFERENCE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

_DENSITY_PROMPT = textwrap.dedent("""\
    You are an expert in polymer science.
    Give the experimental bulk density of amorphous {polymer} at room temperature (300 K).
    
    Respond ONLY with a JSON object — no explanation, no markdown:
    {{
      "density_gcm3": <float, single representative value>,
      "density_range_low": <float>,
      "density_range_high": <float>,
      "source_note": "<brief note about grade/condition, ≤20 words>"
    }}
""")

_TG_PROMPT = textwrap.dedent("""\
    You are an expert in polymer science.
    Give the experimental glass transition temperature (Tg) of amorphous {polymer} in Kelvin.
    
    Respond ONLY with a JSON object — no explanation, no markdown:
    {{
      "tg_K": <float, single representative value>,
      "tg_range_low_K": <float>,
      "tg_range_high_K": <float>,
      "source_note": "<brief note, ≤20 words>"
    }}
""")


def _query_llm_json(llm, prompt: str) -> dict:
    """LLM에 질의하고 JSON 파싱 결과 반환."""
    response = llm.invoke(prompt)
    text = response.content.strip()
    # 마크다운 코드블록 제거
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    return json.loads(text)


def get_reference_density(polymer_name: str, llm) -> dict:
    """
    LLM으로 폴리머 실험 밀도값을 조회합니다.

    Returns
    -------
    dict: {
        density_gcm3        : float   – 대표 밀도 (g/cm³)
        density_range_low   : float
        density_range_high  : float
        source_note         : str
    }
    """
    prompt = _DENSITY_PROMPT.format(polymer=polymer_name)
    data = _query_llm_json(llm, prompt)
    print(f"[Validator] Reference density for '{polymer_name}':")
    print(f"            {data['density_gcm3']:.3f} g/cm³  "
          f"(range: {data['density_range_low']:.3f} – {data['density_range_high']:.3f})")
    print(f"            Note: {data['source_note']}")
    return data


def get_reference_tg(polymer_name: str, llm) -> dict:
    """
    LLM으로 폴리머 실험 유리전이온도(Tg)를 조회합니다 (K).

    Returns
    -------
    dict: {
        tg_K               : float
        tg_range_low_K     : float
        tg_range_high_K    : float
        source_note        : str
    }
    """
    prompt = _TG_PROMPT.format(polymer=polymer_name)
    data = _query_llm_json(llm, prompt)
    print(f"[Validator] Reference Tg for '{polymer_name}':")
    print(f"            {data['tg_K']:.1f} K  "
          f"(range: {data['tg_range_low_K']:.1f} – {data['tg_range_high_K']:.1f} K)")
    print(f"            Note: {data['source_note']}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _pct_error(sim: float, ref: float) -> float:
    return abs(sim - ref) / ref * 100.0


def _verdict(err_pct: float, tol: float) -> str:
    return "✅ PASS" if err_pct <= tol else "❌ FAIL"


def validate_density(
    ref_data: dict,
    aa_density: Optional[float] = None,
    cg_density: Optional[float] = None,
    tol_pct: float = DEFAULT_DENSITY_TOL_PCT,
    polymer_name: str = "",
) -> dict:
    """
    AA/CG 밀도를 참고값과 비교하고 PASS/FAIL 판정을 출력합니다.

    Parameters
    ----------
    ref_data    : get_reference_density()의 반환값
    aa_density  : AA 시뮬레이션 평균 밀도 (g/cm³)
    cg_density  : CG 시뮬레이션 평균 밀도 (g/cm³)
    tol_pct     : PASS 허용 오차 (%, 기본 ±5%)
    polymer_name: 출력용 폴리머 이름

    Returns
    -------
    dict: {aa: {density, error_pct, verdict}, cg: {…}, ref: float}
    """
    ref = ref_data["density_gcm3"]
    rng_lo = ref_data["density_range_low"]
    rng_hi = ref_data["density_range_high"]

    title = f"  Density Validation — {polymer_name}" if polymer_name else "  Density Validation"
    sep = "=" * 55

    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"  {'Refer (LLM)':18s}: {ref:.3f} g/cm³  "
          f"[{rng_lo:.3f} – {rng_hi:.3f}]")
    print(f"  {'Tolerance':18s}: ±{tol_pct:.1f} %")
    print("-" * 55)

    results = {"ref": ref, "ref_range": (rng_lo, rng_hi)}

    for label, density in [("AA", aa_density), ("CG", cg_density)]:
        if density is None:
            continue
        err = _pct_error(density, ref)
        verd = _verdict(err, tol_pct)
        # 실험 범위 내 여부도 표시
        in_range = rng_lo <= density <= rng_hi
        range_note = " (실험 범위 내)" if in_range else " (실험 범위 밖)"
        print(f"  {label+' result':18s}: {density:.3f} g/cm³  "
              f"| err = {err:.1f} %  | {verd}{range_note}")
        results[label.lower()] = {
            "density": density,
            "error_pct": round(err, 2),
            "in_range": in_range,
            "verdict": verd,
        }

    print(sep)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AA ↔ CG SELF-CONSISTENCY COMPARISON  (실험 참고값을 쓰지 않음)
# ─────────────────────────────────────────────────────────────────────────────
# CG 모델이 얼마나 잘 만들어졌는지는 "CG가 원본 AA를 재현하는가"로 판단합니다.
# AA를 기준(reference)으로 두고 CG의 상대 오차를 계산하므로, LLM이 가져오는
# 실험값의 불확실성이 판정에 섞이지 않습니다.

def compare_aa_cg_density(
    aa_density: Optional[float],
    cg_density: Optional[float],
    tol_pct: float = DEFAULT_DENSITY_TOL_PCT,
) -> dict:
    """
    AA 밀도를 기준으로 CG 밀도의 상대 오차를 계산합니다.

        error_pct = |CG - AA| / AA × 100

    Returns
    -------
    dict: {"aa", "cg", "error_pct", "verdict", "tol_pct", "comparable"}
          한쪽이라도 없으면 comparable=False, error_pct/verdict=None.
    """
    out = {"aa": aa_density, "cg": cg_density, "tol_pct": tol_pct,
           "error_pct": None, "verdict": None, "comparable": False}
    if aa_density is None or cg_density is None:
        return out
    if not aa_density:                       # 0 또는 음수 → 상대오차 정의 불가
        out["verdict"] = "⚠️  N/A (AA density is zero)"
        return out
    err = abs(cg_density - aa_density) / abs(aa_density) * 100.0
    out.update({"error_pct": round(err, 2),
                "verdict": _verdict(err, tol_pct),
                "comparable": True})
    return out


def compare_aa_cg_tg(
    aa_tg: Optional[float],
    cg_tg: Optional[float],
    tol_k: float = DEFAULT_TG_TOL_K,
) -> dict:
    """
    AA Tg를 기준으로 CG Tg의 절대 오차(K)를 계산합니다.

        error_K = |CG - AA|

    Returns
    -------
    dict: {"aa", "cg", "error_K", "verdict", "tol_k", "comparable"}
    """
    out = {"aa": aa_tg, "cg": cg_tg, "tol_k": tol_k,
           "error_K": None, "verdict": None, "comparable": False}
    if aa_tg is None or cg_tg is None:
        return out
    err = abs(float(cg_tg) - float(aa_tg))
    out.update({"error_K": round(err, 1),
                "verdict": _verdict(err, tol_k),
                "comparable": True})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tg CALCULATION FROM ANNEALING LOG
# ─────────────────────────────────────────────────────────────────────────────

import re as _re
_header_re = _re.compile(r"^\s*Step\s+", _re.IGNORECASE)


def _parse_thermo_blocks_in_range(
    lines: list[str], i0: int, i1: int
) -> list[pd.DataFrame]:
    """
    lines[i0:i1] 범위 내의 thermo 블록(Step 헤더 + 숫자행)을 파싱.
    여러 블록이 있으면 모두 반환 (reset_timestep 사이마다 하나씩).
    """
    blocks = []
    i = i0
    while i < i1:
        if _header_re.match(lines[i]):
            header = lines[i].strip().split()
            j = i + 1
            data = []
            while j < i1:
                s = lines[j].strip()
                if not s:
                    break
                if s.startswith(("Loop time", "ERROR", "WARNING")):
                    break
                toks = s.split()
                if not toks:
                    break
                try:
                    float(toks[0].replace("D", "E").replace("d", "e"))
                except ValueError:
                    break
                toks = [t.replace("D", "E").replace("d", "e") for t in toks]
                if len(toks) < len(header):
                    break
                data.append(toks[: len(header)])
                j += 1
            if data:
                arr = np.array(data, dtype=float)
                blocks.append(pd.DataFrame(arr, columns=header))
            i = j
        else:
            i += 1
    return blocks


def _get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in m:
            return m[name.lower()]
    return None


def _bin_thermo(
    key: np.ndarray, T: np.ndarray, rho: np.ndarray, nbins: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    아닐링 thermo 데이터를 `key`(보통 Step) 기준으로 구간 평균하여 노이즈를 줄입니다.

    LAMMPS NPT 램프 로그의 순간 Temp/Density는 ±수십 K, ±0.05 g/cm³ 수준으로
    크게 요동치므로, 냉각/가열 진행도(Step)에 따라 묶어 평균하면 깨끗한
    Temp–Density 곡선을 얻을 수 있습니다. 각 bin의 평균 Temp를 x로 사용합니다.

    반환: (T_binned, rho_binned)  — Temp 오름차순 정렬
    """
    order = np.argsort(key)
    key, T, rho = key[order], T[order], rho[order]
    edges = np.linspace(key.min(), key.max(), nbins + 1)
    idx   = np.clip(np.digitize(key, edges) - 1, 0, nbins - 1)

    Tb, rb = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 0:
            Tb.append(float(T[m].mean()))
            rb.append(float(rho[m].mean()))
    Tb, rb = np.asarray(Tb), np.asarray(rb)
    o = np.argsort(Tb)
    return Tb[o], rb[o]


def _fit_segmented_linear(
    T: np.ndarray,
    rho: np.ndarray,
    margin_frac: float = 0.15,
    n_grid: int = 400,
    prefer_physical: bool = True,
) -> dict:
    """
    꺾은점(breakpoint)이 곧 Tg인 '연속' 이중 선형(bilinear) 회귀.

    모델:  rho(T) = c + a_low·min(T−Tg, 0) + a_high·max(T−Tg, 0)
      - Tg 에서 두 직선이 만나도록 연속 조건을 부과 → 외삽으로 인한 비현실적
        교점(데이터 범위 밖) 문제를 원천 차단.
      - Tg 후보를 데이터 범위 '내부'([Tmin+margin, Tmax−margin])에서만 스캔 →
        Tg 는 항상 온도 범위 안에서 나옴 (온도 범위는 런마다 달라도 자동 대응).
      - 고정된 Tg 에 대해 (c, a_low, a_high) 는 선형 최소제곱으로 한 번에 해.

    prefer_physical=True 이면 물리적으로 타당한 해(고온 rubbery 쪽이 더 가파른
    경우, a_high < a_low)를 우선 채택하고, 그런 해가 없으면 전역 SSE 최소해로
    폴백합니다.

    반환: {"Tg", "p1":(a,b), "p2":(a,b), "sse", "physical"}
          p1 = 저온(glassy) 직선, p2 = 고온(rubbery) 직선  (절편 b 형태로 변환)
    """
    order = np.argsort(T)
    T, rho = T[order], rho[order]
    n = len(T)
    if n < 6:
        raise ValueError(f"피팅에 필요한 데이터가 너무 적습니다: n={n}")

    span = float(T.max() - T.min())
    lo   = T.min() + margin_frac * span
    hi   = T.max() - margin_frac * span
    grid = np.linspace(lo, hi, n_grid)

    def _solve(tg: float):
        x1 = np.minimum(T - tg, 0.0)
        x2 = np.maximum(T - tg, 0.0)
        A  = np.column_stack([np.ones_like(T), x1, x2])
        coef, *_ = np.linalg.lstsq(A, rho, rcond=None)
        c, a_low, a_high = coef
        sse = float(np.sum((rho - A @ coef) ** 2))
        return sse, float(c), float(a_low), float(a_high)

    best = None          # 전역 SSE 최소
    best_phys = None     # 물리적 순서를 만족하는 SSE 최소
    for tg in grid:
        sse, c, a_low, a_high = _solve(tg)
        if best is None or sse < best[0]:
            best = (sse, tg, c, a_low, a_high)
        if a_high < a_low and (best_phys is None or sse < best_phys[0]):
            best_phys = (sse, tg, c, a_low, a_high)

    chosen = best_phys if (prefer_physical and best_phys is not None) else best
    sse, tg, c, a_low, a_high = chosen

    # 연속 모델을 일반 y=a*T+b 형태로 변환 (c 는 Tg 에서의 밀도)
    b_low  = c - a_low  * tg
    b_high = c - a_high * tg
    return {
        "Tg":       float(tg),
        "p1":       (a_low,  b_low),    # glassy (low-T)
        "p2":       (a_high, b_high),   # rubbery (high-T)
        "sse":      sse,
        "physical": bool(best_phys is not None and chosen is best_phys),
    }


def calc_tg_from_annealing(
    log_path: str,
    use_cooling: bool = True,
    min_points_per_side: int = 30,   # (구버전 호환용, 현재는 미사용)
    output_dir: str = ".",
    label: str = "",
    show: bool = True,
    dpi: int = 150,
    n_bins: int | None = None,
    margin_frac: float = 0.15,
) -> float:
    """
    아닐링 CG 시뮬레이션 log에서 Tg를 계산합니다.

    방법:
      - log의 '# ----- Temp increase' / '# ----- Temp decrease' 마커로 구간 분리
      - 냉각(기본) 또는 가열 구간 Temp-Density 데이터 추출
      - Step 기준 구간 평균(binning)으로 순간값 노이즈 제거
      - 꺾은점을 데이터 온도 범위 내부로 제한한 '연속' 이중 선형 회귀
        (continuous segmented linear fit) → 꺾은점 = Tg
      - 고온(rubbery) 쪽이 더 가파른 물리적 해를 우선 채택

    Parameters
    ----------
    log_path           : 아닐링 LAMMPS log 경로
    use_cooling        : True(기본) = 냉각 구간 사용, False = 가열 구간 사용
    min_points_per_side: (구버전 호환용 인자, 현재는 사용하지 않음)
    output_dir         : 그래프 저장 폴더
    label              : 파일명 prefix (예: "cg_anneal")
    show               : Jupyter 인라인 표시 여부
    dpi                : 저장 해상도
    n_bins             : 평균 구간 수 (None이면 데이터 양에 맞춰 자동: 12~40)
    margin_frac        : Tg 탐색 시 양끝에서 제외할 범위 비율 (기본 0.15)

    Returns
    -------
    Tg (K, float)
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    print(f"\n[Validator] Tg 계산 시작: {log_path}")

    lines = log_path.read_text(errors="ignore").splitlines()

    def _find_idx(pattern: str) -> int | None:
        for idx, ln in enumerate(lines):
            if pattern in ln:
                return idx
        return None

    idx_up = _find_idx("# ----- Temp increase")
    idx_dn = _find_idx("# ----- Temp decrease")

    if idx_up is None or idx_dn is None:
        raise RuntimeError(
            "log에서 섹션 마커를 찾지 못했습니다.\n"
            f"  Temp increase idx={idx_up}, Temp decrease idx={idx_dn}\n"
            "in.CGTg 로 생성한 아닐링 log인지 확인하세요."
        )
    if idx_dn <= idx_up:
        raise RuntimeError("섹션 순서 오류: Temp decrease가 Temp increase보다 앞에 있습니다.")

    # ── 구간 파싱 ───────────────────────────────────────────────────────────
    up_blocks = _parse_thermo_blocks_in_range(lines, idx_up, idx_dn)

    # Temp decrease ~ 다음 '# -----' 마커 or EOF
    idx_next = next(
        (i for i in range(idx_dn + 1, len(lines)) if lines[i].startswith("# -----")),
        None,
    )
    dn_end   = idx_next if idx_next is not None else len(lines)
    dn_blocks = _parse_thermo_blocks_in_range(lines, idx_dn, dn_end)

    def _merge(blocks: list[pd.DataFrame]) -> pd.DataFrame:
        if not blocks:
            raise RuntimeError("구간에서 thermo 블록을 찾지 못했습니다.")
        return pd.concat(blocks, ignore_index=True) if len(blocks) > 1 else blocks[0].copy()

    df_up = _merge(up_blocks)
    df_dn = _merge(dn_blocks)

    def _extract(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cT    = _get_col(df, ["Temp", "temp"])
        crho  = _get_col(df, ["Density", "density"])
        cstep = _get_col(df, ["Step", "step"])
        if cT is None or crho is None:
            raise RuntimeError(
                f"Temp/Density 컬럼 없음. 현재 헤더: {list(df.columns)}"
            )
        T   = df[cT].to_numpy(dtype=float)
        rho = df[crho].to_numpy(dtype=float)
        # Step이 없으면 행 순서를 진행도 대용으로 사용
        step = (df[cstep].to_numpy(dtype=float) if cstep is not None
                else np.arange(len(df), dtype=float))
        m   = np.isfinite(T) & np.isfinite(rho) & np.isfinite(step)
        return T[m], rho[m], step[m]

    T_up,  rho_up,  step_up  = _extract(df_up)
    T_dn,  rho_dn,  step_dn  = _extract(df_dn)
    if use_cooling:
        T_use, rho_use, step_use = T_dn, rho_dn, step_dn
    else:
        T_use, rho_use, step_use = T_up, rho_up, step_up

    print(f"[Validator] 사용 구간  : {'Temp decrease (cooling)' if use_cooling else 'Temp increase (heating)'}")
    print(f"[Validator] 데이터 포인트: {len(T_use)}")

    # ── 노이즈 제거: Step(진행도) 기준 구간 평균 ───────────────────────────────
    n = len(T_use)
    if n_bins is None:
        nb = int(min(40, max(12, n // 30)))   # 데이터 양에 맞춘 자동 bin 수
    else:
        nb = int(n_bins)
    nb = max(6, min(nb, n))                     # 안전 범위

    T_bin, rho_bin = _bin_thermo(step_use, T_use, rho_use, nb)
    print(f"[Validator] 평균 구간 수 : {nb}  →  binned points: {len(T_bin)}")

    # ── 연속 segmented(이중 선형) 회귀, 꺾은점 = Tg (범위 내부로 제한) ─────────
    fit = _fit_segmented_linear(T_bin, rho_bin, margin_frac=margin_frac)
    (a1, b1), (a2, b2), tg = fit["p1"], fit["p2"], fit["Tg"]

    order      = np.argsort(T_use)
    T_sorted   = T_use[order]
    rho_sorted = rho_use[order]

    Tmin, Tmax = float(T_sorted.min()), float(T_sorted.max())
    Tline = np.linspace(Tmin, Tmax, 300)

    # ── 그래프 ───────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    curve_label = "NPT cooling" if use_cooling else "NPT heating"
    # 원시 순간값(옅게) + 구간 평균(진하게)
    ax.scatter(T_sorted, rho_sorted, s=8, alpha=0.18, color="#4A90D9",
               label=f"{curve_label} (raw)")
    ax.scatter(T_bin, rho_bin, s=34, alpha=0.95, color="#1F5FA8",
               edgecolor="white", linewidth=0.5, zorder=4,
               label="binned mean")

    # 각 직선은 자기 구간에서만 그려 가독성 향상
    Tg = float(tg)
    Tline_lo = np.linspace(Tmin, Tg, 150)
    Tline_hi = np.linspace(Tg, Tmax, 150)
    ax.plot(Tline_lo, a1 * Tline_lo + b1, lw=2.2, color="#E87040",
            label="fit-1 (glassy)")
    ax.plot(Tline_hi, a2 * Tline_hi + b2, lw=2.2, color="#5ABF7C",
            label="fit-2 (rubbery)")

    ax.axvline(Tg, lw=1.8, ls="--", color="orange")
    ax.text(
        Tg, float(np.mean(rho_bin)),
        f"  Tg = {Tg:.1f} K",
        rotation=90, va="center", ha="left", fontsize=10,
        color="darkorange",
    )

    ax.set_xlim(Tmin - 0.02 * (Tmax - Tmin), Tmax + 0.02 * (Tmax - Tmin))
    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(r"Density (g/cm$^3$)", fontsize=12)
    ax.set_title("Glass Transition Temperature (Segmented Linear Fit)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    prefix = f"{label}_" if label else ""
    fpath  = out / f"{prefix}tg_fit.png"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    print(f"[Validator] Plot saved : {fpath}")
    if show:
        plt.show()
    plt.close(fig)

    tg_rounded = round(float(tg), 1)
    print(f"\n[Validator] 데이터 온도 범위 : {Tmin:.1f} – {Tmax:.1f} K")
    print(f"[Validator] 기울기 a_glass / a_rubber : {a1:.3e} / {a2:.3e}")
    print(f"[Validator] 물리적 순서(고온 더 가파름) : {fit['physical']}")
    print(f"[Validator] SSE (min)        : {fit['sse']:.6g}")
    print(f"[Validator] Tg (breakpoint)  : {tg_rounded:.1f} K")

    return tg_rounded



# ─────────────────────────────────────────────────────────────────────────────
# Tg VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_tg(
    ref_data: dict,
    sim_tg: float,
    tol_k: float = DEFAULT_TG_TOL_K,
    polymer_name: str = "",
    label: str = "CG annealing",
) -> dict:
    """
    시뮬레이션 Tg를 참고값과 비교하고 PASS/FAIL 판정을 출력합니다.

    Parameters
    ----------
    ref_data    : get_reference_tg()의 반환값
    sim_tg      : calc_tg_from_annealing()의 반환값 (K)
    tol_k       : PASS 허용 오차 (K, 기본 ±15 K)
    polymer_name: 출력용 폴리머 이름
    label       : 시뮬레이션 구분명

    Returns
    -------
    dict: {tg_K, ref_tg_K, error_K, in_range, verdict}
    """
    ref  = ref_data["tg_K"]
    rng_lo = ref_data["tg_range_low_K"]
    rng_hi = ref_data["tg_range_high_K"]

    err_k    = abs(sim_tg - ref)
    verd     = _verdict(err_k, tol_k)
    in_range = rng_lo <= sim_tg <= rng_hi
    range_note = " (실험 범위 내)" if in_range else " (실험 범위 밖)"

    title = f"  Tg Validation — {polymer_name}" if polymer_name else "  Tg Validation"
    sep   = "=" * 55

    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"  {'Refer (LLM)':18s}: {ref:.1f} K  [{rng_lo:.1f} – {rng_hi:.1f} K]")
    print(f"  {'Tolerance':18s}: ±{tol_k:.1f} K")
    print("-" * 55)
    print(f"  {label+' result':18s}: {sim_tg:.1f} K  "
          f"| err = {err_k:.1f} K  | {verd}{range_note}")
    print(sep)

    return {
        "tg_K":       sim_tg,
        "ref_tg_K":   ref,
        "error_K":    round(err_k, 1),
        "in_range":   in_range,
        "verdict":    verd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONE-STEP DENSITY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    polymer_name: str,
    llm,
    aa_density: Optional[float] = None,
    cg_density: Optional[float] = None,
    tol_pct: float = DEFAULT_DENSITY_TOL_PCT,
) -> dict:
    """
    LLM 참고값 조회 + 밀도 비교를 한 번에 실행합니다.

    Returns
    -------
    dict: {ref_data, density_results}
    """
    ref_data = get_reference_density(polymer_name, llm)
    density_results = validate_density(
        ref_data,
        aa_density=aa_density,
        cg_density=cg_density,
        tol_pct=tol_pct,
        polymer_name=polymer_name,
    )
    return {"ref_data": ref_data, "density_results": density_results}
