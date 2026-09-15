# CGMas (Coarse-Graining Multi-Agent System)

**Joohee Choi**<sup>a†</sup>, **Junhyeong Lee**<sup>b†</sup>, and **Seunghwa Ryu**<sup>a,b,c\*</sup>

<sup>a</sup> Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Republic of Korea
<sup>b</sup> KAIST InnoCORE PRISM-AI Center, Korea Advanced Institute of Science and Technology (KAIST), Republic of Korea
<sup>c</sup> Department of AX, Korea Advanced Institute of Science and Technology (KAIST), Republic of Korea

<sup>†</sup> These authors contributed equally. <sup>\*</sup> Corresponding author: ryush@kaist.ac.kr

![license](https://img.shields.io/badge/license-MIT-green) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Code Overview

Bottom-up coarse-grained (CG) molecular dynamics extends polymer simulation beyond the scales accessible to all-atom (AA) methods, yet deriving a CG force field requires expert knowledge of force-field parametrization and simulation scripting. Large language models (LLMs) offer new possibilities for natural language-driven simulation, but their fluency is not evidence of physical validity, and existing coarse-graining tools take the atomistic topology and the bead mapping as given, so the stages where errors originate remain with the user. **We present CGMas (Coarse-Graining Multi-Agent System), a self-correcting multi-agent framework that transforms a plain-language description of a polymer and its target resolution into a validated coarse-grained model.** The system integrates topology construction, melt equilibration, bead mapping, Boltzmann-inversion potential derivation, and validation against the atomistic reference through structured agent collaboration and a persistent session state. Rather than relying on one-shot code generation, CGMas pairs an LLM reasoning layer with deterministic physics-aware tools and bounded generator–reviewer loops, in which four validators return the specific violated constraint together with its local bonding context so that the agent re-derives each repair from the molecular structure itself (Figure 1). The framework supports homopolymers and copolymers across random, alternating, block, and periodic architectures. **CGMas represents a step toward physics-based, AI-driven automation of molecular-simulation pipelines.**

## Overview of CGMas for automation of polymer coarse-graining

We assessed the robustness of the CGMas system using a twenty-seven-task benchmark suite organized as a five-level difficulty ramp, covering: **hydrocarbons** (4 tasks), **heteroatom-substituted repeat units** (7 tasks), **complex side chains** (9 tasks), **multifunctional repeat units** (3 tasks), and **copolymers** (4 tasks). For the 27 benchmark tasks, CGMas demonstrates the ability to generate physically meaningful CG models without user-supplied topologies or simulation scripts, completing every task, matching the AA equilibrium density to within 5% in 22 of them, and reducing the effective simulation turnaround from 38–88 min to 1 min per task at a reasoning cost of approximately $0.002 per task (Figure 2).

## Requirements

Python ≥ 3.10, LAMMPS (29 Aug 2024 or later, built with the Python module), an MPI runtime, and an LLM API key.

```bash
pip install -r requirements.txt
cp .env.example .env        # then insert your API key
```

## Usage

```bash
jupyter lab CGMas.ipynb
```

```python
# Interactive: CGMas pauses at five checkpoints for user confirmation
run_cgmas("Study CGMD simulation of polyacrylonitrile. Start with all-atomistic simulations.")

# Automated: the full pipeline runs from a single query
run_cgmas_auto("Coarse-grain polystyrene with one backbone bead and one phenyl bead.")
```

All generated files — topology, bead mapping, fitted potentials, LAMMPS inputs and logs, figures, and the validation summary — are written to a timestamped session directory under `output/`.

## Repository

| File | Contents |
|---|---|
| `CGMas.ipynb` | Agent definitions, prompts, and the LangGraph state machine |
| `aa_constructor.py`, `copolymer.py` | OPLS-AA topology construction and melt packing |
| `mapper.py`, `potential.py` | Bead mapping and Boltzmann-inversion potential derivation |
| `simulator.py`, `cg_constructor.py` | LAMMPS input generation and execution |
| `analyzer.py`, `validator.py` | Property extraction and AA-vs-CG validation |
| `docs/MANUAL.md` | Full user manual |
| `examples/` | Runnable case studies, including the benchmark suite |

## Citation

```bibtex
@article{choi2026cgmas,
  title   = {A Multi-Agent Framework for Automated Coarse-Grained Molecular Dynamics of Polymers},
  author  = {Choi, Joohee and Lee, Junhyeong and Ryu, Seunghwa},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026}
}
```

## License

MIT License. See [`LICENSE`](LICENSE).
