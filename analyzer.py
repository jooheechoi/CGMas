"""
analyzer.py
===========
LAMMPS log 분석 도구 for CGMas.

Public functions:
  parse_npt_log(log_path, label)   – NPT 섹션 데이터 파싱
  plot_npt_properties(...)          – Temp / PotEng / Density vs Time 그래프
  calc_equilibrium_density(...)     – 마지막 10% 스텝 평균 밀도 계산
  analyze(...)                      – 위 세 함수를 한 번에 실행
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FS_PER_STEP = 1.0        # timestep = 1 fs (LAMMPS default in real units)
PS_PER_STEP = FS_PER_STEP / 1000.0   # 1 step = 0.001 ps

# thermo 컬럼 이름 (소문자 → pandas 컬럼명)
_COL_MAP = {
    "step":    "step",
    "temp":    "temp",
    "toteng":  "toteng",
    "poteng":  "poteng",
    "kineng":  "kineng",
    "e_mol":   "e_mol",
    "epair":   "epair",
    "press":   "press",
    "vol":     "vol",
    "volume":  "vol",
    "density": "density",
}

_NEEDED = ["step", "temp", "poteng", "density"]


# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_npt_log(log_path: str, label: str = "") -> pd.DataFrame:
    """
    LAMMPS log에서 NPT 섹션 thermo 데이터를 파싱하여 DataFrame 반환.

    반환 컬럼: step, time_ps, temp, poteng, density

    Parameters
    ----------
    log_path : LAMMPS log 파일 경로
    label    : 시뮬레이션 구분 ("aa" / "cg" 등) — DataFrame에 'label' 컬럼 추가
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text()

    # ── NPT 섹션 찾기 ───────────────────────────────────────────────────────
    # "# ----- NPT" 이후의 텍스트만 사용
    npt_match = re.search(r'#\s*-+\s*NPT', text)
    if not npt_match:
        raise ValueError(
            f"'# ----- NPT' 마커를 찾을 수 없습니다: {log_path}\n"
            "log 파일에 NPT 섹션이 있는지 확인하세요."
        )
    npt_text = text[npt_match.start():]

    # ── thermo 헤더 + 데이터 블록 추출 ────────────────────────────────────
    # 헤더 예: "   Step          Temp          TotEng  ..."
    header_match = re.search(
        r'^\s*(Step\s+\S.*?)$',
        npt_text, re.MULTILINE | re.IGNORECASE
    )
    if not header_match:
        raise ValueError("NPT 섹션에서 thermo 헤더를 찾을 수 없습니다.")

    header_line = header_match.group(1)
    col_names_raw = header_line.split()
    col_names = [_COL_MAP.get(c.lower(), c.lower()) for c in col_names_raw]

    # 헤더 이후 데이터 파싱 (숫자로 시작하는 행만)
    data_start = npt_match.start() + header_match.end()
    data_text  = text[data_start:]

    rows: list[list[float]] = []
    for line in data_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # "Loop time" 등 종료 마커
        if re.match(r'^(Loop|WARNING|Per MPI|ERROR)', stripped):
            break
        # 숫자 행만 (첫 토큰이 정수)
        tokens = stripped.split()
        if not tokens or not re.match(r'^\d+$', tokens[0]):
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            continue

    if not rows:
        raise ValueError("NPT 섹션에서 thermo 데이터를 파싱할 수 없습니다.")

    df = pd.DataFrame(rows, columns=col_names[:len(rows[0])])

    # 필요 컬럼 확인
    for col in _NEEDED:
        if col not in df.columns:
            raise ValueError(
                f"필요한 컬럼 '{col}'이(가) log에 없습니다. "
                f"현재 컬럼: {list(df.columns)}"
            )

    # step → ps 변환
    df["time_ps"] = df["step"] * PS_PER_STEP

    # 선택 컬럼만 유지
    keep = ["step", "time_ps", "temp", "poteng", "density"]
    df = df[keep].copy()

    if label:
        df["label"] = label

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

_PROP_META = {
    "temp":    {"ylabel": "Temperature (K)",          "color": "#E87040", "title": "Temperature"},
    "poteng":  {"ylabel": "Potential Energy (kcal/mol)", "color": "#4A90D9", "title": "Potential Energy"},
    "density": {"ylabel": "Density (g/cm³)",           "color": "#5ABF7C", "title": "Density"},
}


def plot_npt_properties(
    *dfs: pd.DataFrame,
    output_dir: str = ".",
    prefix: str = "",
    show: bool = True,
    dpi: int = 150,
) -> list[str]:
    """
    Temp / PotEng / Density vs Time(ps) 그래프를 저장하고 경로 리스트 반환.

    Parameters
    ----------
    *dfs       : parse_npt_log()의 반환값 (여러 개 전달 시 같은 그래프에 겹쳐 그림)
    output_dir : 그래프 저장 폴더
    prefix     : 파일명 앞에 붙일 접두사 (예: "aa_", "cg_")
    show       : Jupyter에서 인라인 표시 여부
    dpi        : 해상도
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []

    for prop, meta in _PROP_META.items():
        fig, ax = plt.subplots(figsize=(8, 4))

        for i, df in enumerate(dfs):
            lbl = df["label"].iloc[0] if "label" in df.columns else f"run{i+1}"
            color = list(_PROP_META.values())[list(_PROP_META.keys()).index(prop)]["color"]
            # 여러 데이터셋이면 색상 순환
            colors = ["#E87040", "#4A90D9", "#5ABF7C", "#9B59B6"]
            ax.plot(
                df["time_ps"], df[prop],
                color=colors[i % len(colors)],
                linewidth=0.8,
                alpha=0.85,
                label=lbl,
            )

        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel(meta["ylabel"], fontsize=12)
        ax.set_title(meta["title"], fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(dfs) > 1 or (len(dfs) == 1 and "label" in dfs[0].columns):
            ax.legend(fontsize=10)

        fig.tight_layout()

        fname = f"{prefix}{prop}_npt.png"
        fpath = out / fname
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        saved.append(str(fpath))
        print(f"[Analyzer] Saved: {fpath}")

        if show:
            plt.show()
        plt.close(fig)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_equilibrium_density(df: pd.DataFrame, last_fraction: float = 0.1) -> float:
    """
    NPT 데이터프레임에서 마지막 last_fraction(기본 10%) 구간의
    평균 밀도를 소수점 셋째 자리까지 반올림하여 반환.

    Parameters
    ----------
    df            : parse_npt_log() 반환 DataFrame
    last_fraction : 평균 구간 비율 (0 < x ≤ 1)
    """
    n_total  = len(df)
    n_last   = max(1, int(n_total * last_fraction))
    tail_df  = df.iloc[-n_last:]
    avg_dens = tail_df["density"].mean()
    return round(avg_dens, 3)


# ─────────────────────────────────────────────────────────────────────────────
# ALL-IN-ONE
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    log_path: str,
    label: str = "",
    output_dir: str = ".",
    show: bool = True,
) -> dict:
    """
    log 파일 하나를 분석하여 그래프 저장 + 평균 밀도 반환.

    Returns
    -------
    dict with keys:
        df          : 파싱된 DataFrame
        plot_paths  : 저장된 그래프 경로 리스트
        density     : 마지막 10% 평균 밀도 (g/cm³, 소수점 3자리)
    """
    print(f"\n{'='*55}")
    print(f"[Analyzer] 분석 시작: {log_path}  label={label!r}")
    print(f"{'='*55}")

    df = parse_npt_log(log_path, label=label)

    n_steps = int(df["step"].max())
    t_total = df["time_ps"].max()
    print(f"[Analyzer] NPT 스텝 수 : {n_steps:,}  ({t_total:.1f} ps)")
    print(f"[Analyzer] thermo 행 수: {len(df):,}")

    prefix = f"{label}_" if label else ""
    plot_paths = plot_npt_properties(df, output_dir=output_dir, prefix=prefix, show=show)

    density = calc_equilibrium_density(df)
    n_last  = max(1, int(len(df) * 0.1))
    print(f"\n[Analyzer] 마지막 10% ({n_last}행) 평균 밀도: {density:.3f} g/cm³")

    return {"df": df, "plot_paths": plot_paths, "density": density}


def compare_density(results: dict[str, dict]) -> None:
    """
    AA / CG 결과를 비교 출력.

    Parameters
    ----------
    results : {"aa": analyze(...), "cg": analyze(...)}
    """
    print("\n" + "="*45)
    print("  밀도 비교 (마지막 10% NPT 평균)")
    print("="*45)
    for lbl, res in results.items():
        print(f"  {lbl.upper():>4s}  :  {res['density']:.3f} g/cm³")
    vals = [r["density"] for r in results.values()]
    if len(vals) == 2:
        diff = abs(vals[0] - vals[1])
        rel  = diff / ((vals[0] + vals[1]) / 2) * 100
        print(f"  차이    :  {diff:.3f} g/cm³  ({rel:.1f}%)")
    print("="*45)
