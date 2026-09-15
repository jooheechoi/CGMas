# CGMas User Manual

Complete reference for installing, configuring, running, and extending CGMas.

For a short description of the framework, see the [README](../README.md). A worked example with real output is in [`examples/polyethylene`](../examples/polyethylene). This manual assumes no prior familiarity with the code.

---

## Contents

1. [How CGMas works](#1-how-cgmas-works)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Running a session](#4-running-a-session)
5. [Writing effective instructions](#5-writing-effective-instructions)
6. [Parameter reference](#6-parameter-reference)
7. [Methodology reference](#7-methodology-reference)
8. [Output files and data formats](#8-output-files-and-data-formats)
9. [Extending CGMas](#9-extending-cgmas)
10. [Troubleshooting](#10-troubleshooting)
11. [Limitations](#11-limitations)
12. [Reproducing the benchmark](#12-reproducing-the-benchmark)

---

## 1. How CGMas works

CGMas is organized as a state machine (LangGraph) whose nodes are of two kinds:

- **Reasoning agents**, backed by an LLM, perform chemical inference, natural-language parsing, and review. Their output is variable and is always validated before use.
- **Deterministic tools**, implemented as Python modules, perform topology assembly, melt construction, mapping, distribution accumulation, potential inversion, and simulation. They produce reproducible results for a fixed input.

All agents are stateless. Every intermediate result is held in a single typed record (`LAMMPSState`) that is threaded through each stage.

### Pipeline stages

| # | Stage | Node(s) | Kind |
|---|---|---|---|
| 1 | Input parsing | `greet`, `copoly_spec` | agent |
| 2 | AA topology generation | `definer` → validators → `aa_constructor` → `aa_review` | agent + tool + agent |
| 3 | AA simulation | `sim_params`, `sim_build`, `aa_run` | tool |
| 4 | Bead mapping | `map_design`, `map_confirm`, `mapper` | agent + tool |
| 5 | Potential derivation | `pot_gate`, `potential`, `pot_review`, `pot_confirm`, `topology` | tool + agent |
| 6 | CG simulation | `cg_init`, `cg_params`, `cg_constructor`, `cg_sim_build`, `cg_run` | tool |
| 7 | Analysis and validation | *T*<sub>g</sub> annealing, validation summary, `plots` | tool + agent |

The compiled graph can be rendered from the notebook:

```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

### Self-correction loops

Three loops return the workflow to an earlier stage when a check fails. Each is bounded so a failing chemistry cannot loop indefinitely.

| Loop | Trigger | Limit | Constant |
|---|---|---|---|
| Topology regeneration | A validator rejects the CRU | 5 attempts | in `definer_agent` |
| AA data-file review | The review agent requests a revision | 2 revisions | `MAX_REVISIONS` (`aa_constructor.py`) |
| CG potential review | The review agent flags a fitted parameter | 2 revisions | `MAX_POTENTIAL_REVISIONS` (notebook) |

When a limit is reached, the current result is retained with a printed warning rather than aborting the session.

### User checkpoints

Five points pause for confirmation in interactive mode. In automated mode they are answered from the parsed query.

| Checkpoint | CGMas asks | You supply |
|---|---|---|
| Polymer identity | `Which polymer would you like to study?` | polymer name, optionally with architecture |
| System parameters | `chain length ..., number of chains ..., density ..., box size ... Confirm?` | accept, or state different values |
| Simulation protocol | annealing cycles, temperatures, equilibration time | accept, or state a different protocol |
| Bead mapping | `How would you like to map the beads?` | resolution description in plain language |
| Potential approval | derived bond, angle, and LJ parameters | accept, or request a type be kept or removed |
| CG run and *T*<sub>g</sub> | `Ready to start CG simulation` / `Ready for Tg annealing` | confirm to proceed |

---

## 2. Installation

### 2.1 Python environment

```bash
git clone https://github.com/jooheechoi/CGMas.git
cd CGMas

conda create -n cgmas python=3.11
conda activate cgmas
pip install -r requirements.txt
```

### 2.2 LAMMPS

CGMas drives LAMMPS through its **Python module**, launched under `mpirun` via a generated `_mpi_runner.py`. A plain `lmp` executable is not sufficient — LAMMPS must be importable as a Python package.

`requirements.txt` installs the unofficial LAMMPS wheels, which are built against MPICH on Linux and macOS and against Microsoft MPI on Windows. **The `mpirun` on your `PATH` must come from the same MPI implementation.** Mixing OpenMPI's `mpirun` with an MPICH-built wheel launches N independent serial runs instead of one parallel run, without raising an error.

To use your own build instead, configure with at least:

```bash
cmake ../cmake -D BUILD_SHARED_LIBS=ON \
               -D PKG_PYTHON=ON \
               -D PKG_MOLECULE=ON \
               -D PKG_KSPACE=ON
make -j && make install-python
```

### 2.3 Verification

Run these in the same interpreter you will use for the notebook:

```bash
python -c "import lammps; print(lammps.lammps().version())"
mpirun --version
```

`import lammps` must succeed. If it does not, add the directory containing `liblammps.so` to `LD_LIBRARY_PATH` and the Python wrapper to `PYTHONPATH`.

### 2.4 Required data files

CGMas reads two sets of files from the repository root. Confirm both are present before the first run.

```
opls-aa/
├── atom.txt         # atom types, masses, charges, LJ epsilon and sigma
├── bond.txt         # bond types, r0, k
├── angle.txt        # angle types, theta0, k
└── dihedral.txt     # dihedral types, Fourier coefficients

templates/
├── in.AAEQ          # all-atom equilibration template
├── in.CGEQ          # coarse-grained equilibration template
├── in.AATg          # all-atom cooling-scan template
└── in.CGTg          # coarse-grained cooling-scan template
```

`atom.txt` is read twice: once by `load_opls()` to supply numerical parameters to the constructor, and once in full as text, where it is inserted into the topology agent's system prompt. The agent selects among the entries of this table and introduces no types of its own, so the table bounds the chemical coverage of the framework.

The `in.*` templates define the LAMMPS `pair_style`, cutoffs, neighbour settings, thermostat and barostat parameters, and dump settings. Changing a protocol globally is done here, not in the Python code.

---

## 3. Configuration

### 3.1 API key

The notebook loads credentials in its first cell:

```python
load_dotenv("openai.env")
```

Copy the template and insert your key:

```bash
cp openai.env.example openai.env
```

```dotenv
OPENAI_API_KEY=sk-...
```

`openai.env` is listed in `.gitignore` and must never be committed. If you prefer the conventional `.env`, rename the file and change the call to `load_dotenv()` with no argument.

### 3.2 Reasoning backbone

Set in cell 2 of the notebook:

```python
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
```

Any LangChain chat model can be substituted. For example, with `langchain-anthropic` installed and `ANTHROPIC_API_KEY` set:

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
```

`temperature=0` is recommended. The topology agent selects among tabulated parameters rather than proposing values, so sampling adds self-correction cycles and token cost without improving the physical output.

Five backbones were benchmarked on the same task. All completed the pipeline and passed validation, with density errors in a narrow band, while monetary cost varied by a factor of roughly 25. `gpt-5.4-mini` is the default because it was both the least expensive and the most accurate of those tested. See [`examples/polyethylene`](../examples/polyethylene) for the comparison.

### 3.3 Token accounting

A `TokenCostTracker` callback records the number of LLM calls and input and output tokens, and writes `token_usage.json` and `token_cost.txt` into the session directory. Per-million-token prices are in `TOKEN_PRICES_USD_PER_1M` in the notebook; a backbone missing from that table reports tokens but no dollar figure.

### 3.4 MPI processes

The number of MPI ranks is passed to `run_lammps(..., n_cores=N)`. The published results used `n_cores=12` on a 20-core workstation. Set `n_cores=1` to run serially if no matching MPI runtime is available.

---

## 4. Running a session

### 4.1 Starting

```bash
jupyter lab CGMas.ipynb
```

Execute the cells in order. Cells 0–17 define the state schema, agent prompts, graph nodes, and compile the workflow; cell 18 renders the graph; **cell 19 defines `run_cgmas` and `run_cgmas_auto`**. Nothing runs until you call one of them.

Each call creates a new session directory `output/YYMMDD_NNN/` and a unique `thread_id`. **Sessions cannot be resumed.** Re-using a `thread_id` causes the LangGraph checkpointer to replay earlier messages, which makes the duplicate-suppression guard skip interrupt prompts; the entry points therefore generate a fresh id on every call. If a run fails, start a new one.

### 4.2 Automated mode

```python
result = run_cgmas_auto(
    "Run simulation of polyethylene aa and cg. Set 1 bead per monomer."
)
```

The entire pipeline — AA construction and equilibration, mapping, potential derivation, CG equilibration, *T*<sub>g</sub> scan, validation, plots — runs from a single query with no further input. Checkpoints are answered from the parsed query, falling back to module defaults. Use this for batch runs and benchmark reproduction.

On entry the function reports the session directory and what it extracted from the query:

```
Session directory: /path/to/CGMas/output/260604_001
[AUTO] Running full pipeline from a single query -- no further input needed.

[AUTO] Parsed directives: {'aa_equil_ps': 1000, 'cg_equil_ps': 1000}
```

**What is parsed.** `parse_directives()` extracts exactly six settings, and only when they are stated explicitly:

| Directive | Set by wording such as |
|---|---|
| `n_monomers` | `chain length 20`, `20 monomers per chain`, `degree of polymerization 20`, `a 20-mer` |
| `n_chains` | `10 chains`, `number of chains 10` |
| `aa_density` | `density 0.85 for aa` |
| `cg_density` | `density 0.85 for cg` |
| `aa_equil_ps` | `EQ time 1 ns for aa` (ns are converted to ps) |
| `cg_equil_ps` | `EQ time 500 ps for cg` |

A bare `density 0.85` or `EQ time 1 ns`, with no `aa`/`cg` qualifier, sets both members of the pair. Anything absent falls back to the module default; the parser never guesses.

**Mapping phrases are deliberately not read as chain length.** `1 bead per monomer`, `1 bead = 1 monomer`, and `map each monomer to one bead` describe the CG resolution and leave `n_monomers` unset. This distinction matters because both quantities can appear in the same sentence.

**Not parsed: temperature and annealing cycles.** These are not among the six directives, so a query stating `equilibrated at 350 K` is silently ignored in automated mode. To vary them, use interactive mode or edit `DEFAULT_T_INIT`, `DEFAULT_T_MAX`, and `DEFAULT_N_ANNEAL` in `simulator.py`.

At the end of the run a token report is printed and written to the session directory. Errors are caught rather than raised: the report is still emitted and the session directory returned, so a failed run leaves a complete log to read.

### 4.3 Interactive mode

```python
session_dir = run_cgmas(
    "Study CGMD simulation of polyethylene. Start with all-atomistic simulations."
)
```

CGMas prints its reasoning and pauses at each checkpoint with a `You:` prompt. Use this when you want to change a decision mid-run, or to set a temperature or annealing protocol, which automated mode cannot read from the query.

### 4.4 Console logging

The notebook's last cell mirrors stdout into `cgmas_log.txt` with a `Tee` wrapper. **This file is written to the notebook's working directory, not into the session directory.** Move it into the session folder afterwards if you want the session to be self-contained. `chat_log.json`, inside the session directory, records every agent message and reply regardless.

---

## 5. Writing effective instructions

### 5.1 Polymer names

Give a common or IUPAC name. Both `polystyrene` and `poly(methyl methacrylate)` work; abbreviations such as `PMMA` are usually resolved but are less reliable than the full name. Tacticity is not modelled, and specifying it has no effect.

### 5.2 Copolymers

State the architecture and the composition in the same sentence. The parsing agent resolves them into an architecture keyword (`random`, `alternating`, `block`, `periodic`, `graft`), mole fractions, block lengths, or a repeat pattern.

| Intent | Phrasing that works |
|---|---|
| Random, equimolar | `random copolymer of ethylene and propylene, 50:50` |
| Alternating | `alternating copolymer of styrene and maleic anhydride` |
| Block | `block copolymer of styrene and butadiene, blocks of 5 and 5` |
| Periodic | `periodic copolymer of ethylene and vinyl alcohol with pattern AAB` |

The topology agent applies the full self-correction pipeline independently to each comonomer before the sequences are generated.

### 5.3 Bead resolution

The mapping agent converts your description into a non-overlapping partition of the repeat unit. Be explicit about **how many beads** and **which atoms belong to each**.

| Intent | Phrasing that works |
|---|---|
| One bead per repeat unit | `1 monomer = 1 bead` |
| One bead per two repeat units | `2 monomers = 1 bead` |
| Backbone plus side group | `two bead types: backbone and the ester side group` |
| Named chemical groups | `separate beads for the CH2-CH backbone and the phenyl ring` |
| Explicit atom groups | `map NH-CH2-CH2 to bead type A, CH2-CH2 to bead type B, and CH2-C(=O) to bead type C` |

Phrasings to avoid: `coarse-grain it reasonably`, `use a standard mapping`, `make it as coarse as possible`. These leave the partition undetermined and the agent will pick one without indicating that it guessed. If the returned mapping is not what you intended, reject it at the checkpoint and restate the grouping atom by atom, as in the last row above.

A bead may span more than one repeat unit (`repeating_monomers` > 1), but every atom of the repeat unit must belong to exactly one bead.

### 5.4 Simulation protocol

**In interactive mode**, state the annealing cycles, the temperature range, and the equilibration time in one sentence at the protocol checkpoint:

```
Run 2 annealing cycles from 300 K to 500 K, then simulate for 2000 ps at 300 K.
```

Temperatures given in the wrong order are corrected automatically, and a notice is printed.

**In automated mode**, only the equilibration time is read from the query (`aa_equil_ps`, `cg_equil_ps`). Temperature and cycle count must be changed in `simulator.py`. See [§4.2](#42-automated-mode).

---

## 6. Parameter reference

### 6.1 System construction — `aa_constructor.py`

| Constant | Meaning | Default |
|---|---|---|
| `DEFAULT_N_CHAINS` | chains per system, *N*<sub>c</sub> | 20 |
| `DEFAULT_N_MONOMERS` | repeat units per chain, *N*<sub>m</sub> | 10 |
| `DEFAULT_DENSITY` | initial packing density | 0.9 g cm<sup>−3</sup> |
| `DEFAULT_SEED` | random seed for coordinate generation | 42 |
| `MAX_REVISIONS` | AA review retry limit | 2 |

The seed is fixed by default, so a repeated run with the same CRU produces identical coordinates. Change it to sample independent configurations, or to escape a packing that fails on the first integration step.

### 6.2 All-atom simulation — `simulator.py`

| Constant | Meaning | Default |
|---|---|---|
| `DEFAULT_T_INIT` | operating temperature | 300 K |
| `DEFAULT_T_MAX` | annealing peak temperature | 500 K |
| `DEFAULT_N_ANNEAL` | annealing cycles | 1 |
| `DEFAULT_EQUIL_PS` | final NPT equilibration time | 1000 ps |
| `DEFAULT_ANNEAL_SPAN` | fallback *T*<sub>max</sub> − *T*<sub>init</sub> when only *T*<sub>init</sub> is given | 200 K |
| `STEPS_PER_PS` | 1 fs time step | 1000 |
| `DEFAULT_TG_MARGIN_K` | half-width of the cooling-scan window about the reference *T*<sub>g</sub> | 120 K |
| `_TINI_FLOOR` | lower bound on the scan start temperature | 80 K |

### 6.3 Coarse-grained simulation — `cg_constructor.py`

| Constant | Default |
|---|---|
| `DEFAULT_T_INIT_CG` | 300 K |
| `DEFAULT_T_MAX_CG` | 500 K |
| `DEFAULT_N_ANNEAL_CG` | 1 |
| `DEFAULT_EQUIL_PS_CG` | 1000 ps |
| `DEFAULT_ANNEAL_SPAN_CG` | 200 K |

### 6.4 Potential derivation — `potential.py`

Passed to `compute_cg_potentials_multi()`.

| Argument | Meaning | Default |
|---|---|---|
| `temperature` | *T* used in the Boltzmann inversion | AA run temperature |
| `frame_stride` | trajectory frames skipped between samples | 1 |
| `bond_bins` | bond-length histogram bins over 0–15 Å | 200 |
| `angle_bins` | angle histogram bins over 0–180° | 180 |
| `rdf_bins` | RDF histogram bins | 500 |
| `rdf_frac` | RDF cutoff as a fraction of the shortest box edge | 0.35 |
| `rdf_exclude_depth` | bonded neighbours excluded from the RDF; 3 excludes 1-2, 1-3, 1-4 | 3 |
| `energy_matching` | rescale each LJ ε to the AA intermolecular cohesive energy per bead | `True` |
| `epsilon_scale` | fixed extra multiplier on ε, used only when energy matching is off | 1.0 |
| `cg_pair_cutoff` | cutoff used in the energy-matching integral | 10.0 Å |

Increase `bond_bins` or `rdf_bins` for noisy distributions; increase the equilibration time or the system size instead if a distribution is under-sampled, since finer bins add no information.

### 6.5 Validation — `validator.py`

| Constant | Meaning | Default |
|---|---|---|
| `DEFAULT_DENSITY_TOL_PCT` | pass band on the CG-vs-AA density error | 5.0 % |
| `DEFAULT_TG_TOL_K` | pass band on the CG-vs-AA *T*<sub>g</sub> difference | 15.0 K |

### 6.6 Overriding

Four routes, in increasing order of permanence:

1. **In an automated query.** Only the six directives of [§4.2](#42-automated-mode).
2. **At a checkpoint, in interactive mode.** Anything the node asks about, including temperature and annealing protocol. Applies to that session only.
3. **In the notebook.** Pass an explicit argument to the tool call inside the corresponding node.
4. **In the module.** Edit the `DEFAULT_*` constant. Applies to every subsequent session.

`pair_style`, cutoffs, and thermostat and barostat settings live in the `templates/in.*` files, not in the Python code.

---

## 7. Methodology reference

### 7.1 Topology generation and validation

Given a polymer name, the topology agent returns a JSON description of the constitutional repeat unit (CRU): the smallest fragment whose repetition reproduces the chain. For each atom it assigns an OPLS-AA type and partial charge matched to the local bonding environment, lists the intramolecular bonds, and designates the head and tail atoms that connect to adjacent repeat units. The raw output is echoed to the console between `--- Formatter output ---` and `--- Formatter done ---`.

Deterministic post-processing then runs before the validators and prints each change:

```
  [Auto-add H] local_id=2 CT_2: added 2 HC_1 (had 0 H, needed 2)
  [Auto-add ring/vinyl H] local_id=1 N_2: added H_2 (N H)
  [Auto-neutralize residual] spread residual net -0.2000 e over 8 heavy atoms
[Formatter] Charge OK: net = +0.0000 e  |  topology OK
```

Hydrogens omitted by the agent are added from the valence rule `4 − N_heavy − N_inter`, where `N_inter` is 1 for the head and tail atoms. An `[Auto-neutralize residual]` line means the OPLS table contained no exactly neutral typing for that repeat unit and the residual charge was spread over the heavy atoms; treat it as a warning that the charge distribution, and therefore the cohesive energy, is approximate for that chemistry.

Four validators then run. A failing validator returns the quantitative discrepancy together with the relevant local bonding context — not a corrected assignment — so that the agent re-derives the repair from the structure itself.

| Validator | Check | Feedback on failure |
|---|---|---|
| Charge neutrality | \|Σ*q*<sub>i</sub>\| ≤ 0.01 e over the CRU | numerical charge error, plus a per-atom charge table annotated with bonded neighbours |
| Hydrogen typing | alkene H on sp² alkene C, aromatic H on aromatic C, alkane H on sp³ C | expected bonded-carbon type and acceptable alternatives |
| Chain-end valence | H count on a carbon equals 4 − *N*<sub>heavy</sub> − *N*<sub>inter</sub> | excess or missing hydrogens, per atom |
| Rule-based post-processing | recurring heteroatom and halogen misassignments (hydroxyl vs. ether oxygen, ether carbons, alkene carbon subtypes, chlorine keys) | corrected in place |

The agent is re-invoked for a targeted repair for up to five cycles. After the constructor writes the LAMMPS data file, a separate review agent inspects the file against the CRU definition for format integrity, force-field completeness, chemical consistency of the atom typing, and head–tail connectivity. Charge neutrality and file completeness are established upstream and excluded from this review.

### 7.2 All-atom construction

Chains are built by breadth-first traversal of the intramolecular bond graph with Z-matrix placement. Bond lengths and angles come from the OPLS-AA tables; torsions are sampled uniformly from [0, 2π) to give amorphous conformations. Rings, detected as back-edges in the traversal, are placed as regular polygons oriented by a random rotation about the ring-attachment bond, which handles aromatic and other cyclic groups at arbitrary side-chain depth.

Chains are placed sequentially into a cubic periodic box of side

    L = ( N_c · N_m · M_m / ( N_A · rho ) )^(1/3)

where *M*<sub>m</sub> is the repeat-unit molar mass — the fraction-weighted average over comonomers for a copolymer. Steric overlaps are avoided by adaptive overlap rejection: for each newly placed atom the torsion is resampled, up to a few hundred attempts, until the atom clears all previously placed atoms. The minimum separation is set per atom type, smaller for hydrogens than for heavier atoms, so that tight but physical contacts are retained while genuine clashes are rejected.

### 7.3 Equilibration

Conjugate-gradient energy minimization, then an annealing protocol in which the temperature is ramped from *T*<sub>init</sub> to *T*<sub>max</sub> and back over *N*<sub>anneal</sub> cycles, each ramp segment 100 ps by default. The system is then equilibrated in NPT with a Nosé–Hoover thermostat and barostat for *t*<sub>equil</sub>.

Interactions use OPLS-AA with geometric mixing for both ε and σ. Equations of motion are integrated with velocity-Verlet at a 1 fs time step under periodic boundary conditions in all three directions.

Equilibration is assessed from the convergence of temperature, density, and potential energy over the final portion of the NPT trajectory, which is recorded in `lammps_aa.log`. **Inspect it before trusting any downstream result** — the accumulated distributions inherit any residual drift of the reference trajectory.

### 7.4 Mapping

For every trajectory frame, the centre of mass of each bead is computed as

    r_COM = sum_i ( m_i · r_i ) / sum_i m_i

with atom positions unwrapped relative to a reference atom of the bead, so that the COM is evaluated consistently across periodic boundaries. Bead masses are the summed atomic masses of the constituent atoms.

### 7.5 Potential derivation

Bonded distributions and the pair RDF are accumulated from the mapped trajectory and Boltzmann-inverted:

    U_bond(l)      = -kB·T · ln[ P(l) / l^2 ]
    U_angle(theta) = -kB·T · ln[ P(theta) / sin(theta) ]
    U_nb(r)        = -kB·T · ln[ g(r) ]

The *l*² and sin θ terms account for the configurational degeneracy at fixed bond length and fixed angle. The inverted curves are fitted to harmonic bond and angle forms and to a 12-6 Lennard-Jones pair form. Each sampled distribution is written to its own CSV file in the session directory.

**Energy matching.** Boltzmann inversion of *g*(*r*) returns the potential of mean force, whose well depth is far shallower than the effective pair interaction required to hold a melt together; used directly, the CG system expands without bound. Each LJ ε is therefore rescaled so that the CG pair potential reproduces the all-atom intermolecular cohesive energy per bead, integrated to `cg_pair_cutoff` at the AA equilibrium density. This is enabled by default (`energy_matching=True`) and applies to every derived pair type.

**Review.** A review agent cross-checks the fitted set against the enumerated interaction types. Types present in the topology but absent from the results are added and recomputed. Angle equilibria above 180°, and entries with both a vanishing force constant and a vanishing equilibrium value — indicating the type was never sampled — are removed. Bond equilibrium lengths below 1.5 Å are flagged as unphysical for beads representing several atoms. A vanishing pair well depth is retained but reported as insufficiently sampled. The backbone–backbone–backbone angle is restored if missing. Interaction types you specified explicitly are protected from automatic removal.

### 7.6 Glass transition

The scan window is set per polymer rather than fixed. A validation agent is first asked for the experimental *T*<sub>g</sub> of the amorphous polymer, returning a representative value and a range. The cooling trajectory then runs over

    T_start = max( 80 K, T_ref - 120 K )
    T_end   = T_ref + 120 K

so that the transition falls near the middle of the sampled range. The reference value and its source note are printed by the validator; check them, because this is the one place where an LLM lookup enters the physical protocol rather than the chemical reasoning, and an inaccurate reference shifts the window rather than the result.

Two straight lines are fitted to the binned ρ(*T*) data; the breakpoint minimizing the total squared residual of the two segments is retained and the intersection reported as *T*<sub>g</sub>. The fit is accepted only when the high-temperature branch is steeper than the low-temperature branch, as expected for the rubbery and glassy states. The same protocol and chain length are used for the AA and CG systems, so the comparison isolates the effect of coarse-graining.

### 7.7 Validation criteria

Equilibrium densities are averaged over the final third of the NPT trajectory. A task passes when

    | ( rho_CG - rho_AA ) / rho_AA | x 100  <=  5 %

*T*<sub>g</sub> is reported as a secondary quantity, with agreement within 15 K taken as acceptable. The reference is always the AA system produced under the same protocol, so the residual deviation reflects the coarse-graining step rather than a difference in the reference state.

---

## 8. Output files and data formats

### 8.1 Session directory

Each run creates `output/YYMMDD_NNN/`, where `NNN` increments within a day. Most file names carry the polymer name as a prefix or suffix; `<polymer>` below is the polymer name with non-word characters replaced by underscores.

| File | Contents |
|---|---|
| `chat_log.json` | every agent message and user reply, in order |
| `<polymer>_ini.data` | packed all-atom LAMMPS data file |
| `in.<polymer>_AAEQ` | generated AA equilibration script |
| `lammps_aa.log` | AA LAMMPS log — density, temperature, potential energy vs. time |
| `AAEQfin.data` | equilibrated all-atom configuration |
| `aa_fin.lammpstrj` | AA trajectory used for mapping and distribution accumulation |
| `<polymer>_bead.data` | mapped CG configuration |
| `cg_bond_<i>_<j>.csv` | sampled bond-length distribution for bead-type pair (*i*, *j*) |
| `cg_angle_<i>_<j>.csv` | sampled angle distribution |
| `cg_rdf_<i>_<j>.csv` | sampled radial distribution function |
| `<polymer>_cg_ini.data` | CG initial configuration |
| `in.<polymer>_CGEQ` | generated CG equilibration script — **contains the derived coefficients** |
| `lammps_cg.log` | CG LAMMPS log |
| `CGEQfin.data` | equilibrated CG configuration |
| `in.<polymer>_CGTg`, `lammps_tg.log` | cooling-scan script and log |
| `token_usage.json`, `token_cost.txt` | LLM call count, token counts, estimated cost |
| `*_dist.png`, `*tg_fit.png`, `aa_*.png`, `cg_*.png` | figures, when the `plots` node runs |

One CSV triple is written per interaction type, so a one-bead mapping produces three files and a multi-bead mapping proportionally more.

`chat_log.json` is the reproducibility record: it contains every decision the reasoning layer made, including the validator feedback that triggered each correction cycle. Include it when reporting a problem.

`cgmas_log.txt` is written to the notebook's working directory, not here. See [§4.4](#44-console-logging).

### 8.2 Where the derived CG parameters are

**There is no separate parameter file.** The fitted values are written directly into the generated CG input script as LAMMPS coefficient commands:

```
bond_coeff   1  <k>  <r0>
angle_coeff  1  <k>  <theta0>
pair_coeff   1 1  <epsilon>  <sigma>
```

Read them from `in.<polymer>_CGEQ`. Units are kcal mol<sup>−1</sup> Å<sup>−2</sup> for bond *k*, kcal mol<sup>−1</sup> rad<sup>−2</sup> for angle *k*, Å for *r*<sub>0</sub> and *σ*, degrees for *θ*<sub>0</sub>, and kcal mol<sup>−1</sup> for ε. The same values appear in the potential-review exchange in `chat_log.json`, keyed by bead-type tuple.

### 8.3 `poly_def` — constitutional repeat unit

An in-memory structure, not a file; it is recorded in `chat_log.json`.

```json
{
  "polymer_name": "polycaprolactam",
  "description":  "Aliphatic polyamide repeat unit representing nylon-6.",
  "monomer_atoms": [
    { "local_id": 1, "opls_id": "N_2",  "charge": -0.50, "comment": "secondary amide N" },
    { "local_id": 2, "opls_id": "CT_2", "charge": -0.12, "comment": "alkane CH2 alpha to carbonyl" }
  ],
  "intra_bonds": [[1, 2], [2, 3]],
  "head_atom": 1,
  "tail_atom": 7,
  "notes": "Connection points to adjacent repeat units are atoms 1 and 7."
}
```

`local_id` is 1-based within the repeat unit. `head_atom` and `tail_atom` are the attachment points to the preceding and following repeat units; each carries one bond to the adjacent CRU, which reduces its free valence for intra-CRU hydrogens by one. Hydrogens need not be listed — the post-processor adds them from the valence rule and reports each addition.

### 8.4 `bead_def` — mapping definition

Also in-memory, recorded in `chat_log.json`.

```json
{
  "description": "Backbone bead and nitrile side-group bead.",
  "bead_types": [
    {
      "bead_type_id": 1,
      "name": "backbone",
      "local_ids_in_monomer": [1, 2, 3, 4],
      "repeating_monomers": 1
    },
    {
      "bead_type_id": 2,
      "name": "nitrile",
      "local_ids_in_monomer": [5, 6],
      "repeating_monomers": 1
    }
  ]
}
```

For copolymers each bead type additionally carries `monomer_type`, naming the comonomer it maps. `local_ids_in_monomer` must be 1-based, non-overlapping, and must together cover every atom of the repeat unit. `bead_type_id` is the index that appears in the CSV file names and in the `pair_coeff` lines.

---

## 9. Extending CGMas

### 9.1 Adding OPLS-AA atom types

The topology agent selects among the entries of `opls-aa/atom.txt` and introduces none of its own, so extending coverage means extending the table.

1. Add a row to `atom.txt` with a unique key, mass, partial charge, LJ ε and σ, and a `comment` describing the chemical environment. **The comment matters**: it is the only description the agent reads when matching an atom to a type, so state the hybridization, the neighbouring atoms, and the functional-group context explicitly.
2. Add the corresponding bond, angle, and dihedral entries to the other three files. Missing bonded parameters fall back to the element-pair defaults in `aa_constructor.py`, which are approximate and should not be relied on for a published result.
3. If the new type needs a validator rule — for instance a hydrogen subtype with a restricted set of acceptable bonded carbons — add it to the hydrogen-typing validator.
4. Test on a polymer whose repeat unit contains the new type, and confirm that no `[Auto-neutralize residual]` line appears, before using it in a benchmark.

### 9.2 Substituting the reasoning backbone

Replace the `ChatOpenAI(...)` line in cell 2 as in [§3.2](#32-reasoning-backbone). Nothing else needs to change: every agent receives its schema in the system prompt and every output is validated downstream, so the framework does not depend on provider-specific features such as structured-output modes. Add the new model to `TOKEN_PRICES_USD_PER_1M` for cost accounting.

### 9.3 Copolymer architectures

`copolymer.py` generates the chain sequences. `generate_sequence()` implements random, alternating, block, and periodic architectures from mole fractions, block lengths, or a repeat pattern; `graft_branch_positions()` handles graft placement. A new architecture is added by extending `generate_sequence()` and adding the corresponding keyword to the parsing prompt in `copolymer_spec_node`.

### 9.4 Alternative inversion schemes

Only direct Boltzmann inversion with energy matching is implemented. Iterative Boltzmann inversion or force matching would replace the single-pass fit in `compute_cg_potentials_multi()` with a loop that re-runs the CG simulation and updates the potential from the discrepancy between the CG and target distributions. The mapping, topology, and validation stages are independent of this choice and would not need modification.

---

## 10. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `ModuleNotFoundError: No module named 'lammps'` | LAMMPS not installed as a Python package, or not on `PYTHONPATH`. See [§2.2](#22-lammps). |
| Runs finish suspiciously fast and `lammps_aa.log` shows one process | `mpirun` and the LAMMPS build come from different MPI implementations, so N serial runs were launched. See [§2.2](#22-lammps). |
| `FileNotFoundError: .../opls-aa/atom.txt` | The parameter directory is missing. See [§2.4](#24-required-data-files). |
| `AuthenticationError` from the provider | `openai.env` missing, misnamed, or not loaded. Confirm the file is in the repository root and restart the kernel. |
| `MPIAbortException: 'ERROR on proc 2: Bond atoms 3161 3162 missing on proc 2 at step 1'` | The packed configuration contained a contact too close to survive the first integration step. Re-run with a different `DEFAULT_SEED`, or lower `DEFAULT_DENSITY` so the initial box is looser. |
| Topology regeneration exhausts all five attempts | The chemistry is probably outside the supplied OPLS-AA table. Read `chat_log.json` to identify the failing validator, then extend `atom.txt` ([§9.1](#91-adding-opls-aa-atom-types)) or supply a corrected CRU at the checkpoint. |
| `[Auto-neutralize residual]` appears | The table had no exactly neutral typing for the repeat unit. The run continues, but the charge distribution is approximate; expect a larger density error. |
| `No equilibration trajectory found` | The AA run did not produce `aa_fin.lammpstrj`. Check `lammps_aa.log` for the LAMMPS error that aborted it. |
| A pair type reports a vanishing well depth | That bead pair was insufficiently sampled. Increase the equilibration time, *N*<sub>c</sub>, or *N*<sub>m</sub>. |
| CG melt expands or evaporates | Energy matching was disabled, or `epsilon_scale` was set too low. Restore `energy_matching=True`. |
| Bond equilibrium length flagged below 1.5 Å | The mapping probably places two beads on atoms that are not separated along the chain. Restate the partition. |
| CG density deviates by more than 5 % | Check the AA reference first: the traces in `lammps_aa.log` must be stationary before mapping. Polar backbones and bulky side groups are the known weak cases ([§11](#11-limitations)). |
| *T*<sub>g</sub> fit rejected | The cooling scan did not produce two branches of the expected slope ordering. Check the reference *T*<sub>g</sub> the validator printed; if it is far from the literature value the window was misplaced. |
| A temperature stated in an automated query had no effect | Temperature is not one of the six parsed directives. Use interactive mode or edit `simulator.py`. See [§4.2](#42-automated-mode). |

---

## 11. Limitations

- **Chemical coverage is bounded by the supplied table.** Every atom-type assignment is traceable to a tabulated force field, and none is invented by the language model. Exotic functional groups, metal-containing repeat units, and charged monomers such as polyelectrolytes fall outside the present OPLS-AA table.
- **Direct Boltzmann inversion with energy matching** reproduces the sampled distributions and the cohesive energy by construction, but does not self-consistently optimize the pair correlation, so structural agreement is not guaranteed wherever the density is matched. Iterative schemes are not implemented.
- **Known weak cases.** In the benchmark, the largest density deviations occurred for repeat units carrying oxygen in the backbone or in an ester linkage, where the partial charges of the oxygen and its neighbouring carbons balance against one another so that an assignment satisfying overall neutrality can still distribute charge incorrectly. Styrenic repeat units also exceeded the band narrowly, placing a phenyl ring in a bead of unusually large collision diameter.
- **The validation criterion is bulk density.** A criterion that suffices for a polymer melt does not carry over to an interface, a mechanical response, or a phase transition. Extending CGMas to those targets requires its own basis for evaluation.
- **The reasoning layer is not strictly deterministic** even at `temperature=0`. The number of self-correction cycles, and therefore the token cost, varies between runs. The physical output does not: density error reproduced to within 0.03 percentage points across repeated runs, while cost varied by up to 26 % for the same task.
- **Sessions cannot be resumed.** A failed run must be restarted from the beginning.
- **Tacticity, molecular-weight distribution, and branching beyond the graft architecture** are not modelled. All chains in a system have the same length.

---

## 12. Reproducing the benchmark

The benchmark comprises 27 tasks in a five-level difficulty ramp: hydrocarbons (4), heteroatom-substituted repeat units (7), complex side chains (9), multifunctional repeat units (3), and copolymers (4). Every task was run with the default construction and equilibration parameters, so each is a single `run_cgmas_auto()` call of the form

```python
run_cgmas_auto(
    "Run simulation of <polymer> aa and cg. <bead mapping description>."
)
```

Levels 1 and 2 use one bead per repeat unit, so the mapping clause is `Set 1 bead per monomer`. Levels 3 to 5 need two or more bead types; state the partition explicitly, as in [§5.3](#53-bead-resolution).

Three metrics are recorded per task: whether it completed and satisfied the density criterion within the iteration limits, the density error of completed tasks, and the cost in wall-clock simulation time, LLM calls, tokens, and dollars. The first two are read from `lammps_aa.log` and `lammps_cg.log`, the last from `token_cost.txt`. On the reference workstation (Intel Core i7-14700K, 12 MPI processes), the AA reference runs take 38–88 min per task and the CG runs about 1 min.

Exact agreement with the published numbers is not expected for token counts and cost, which depend on the number of self-correction cycles and therefore on the backbone and on run-to-run variation. The density and *T*<sub>g</sub> results are reproducible to the precision quoted in the paper, provided `DEFAULT_SEED` is unchanged.

A fully worked single task, with its session directory and the two robustness tables, is in [`examples/polyethylene`](../examples/polyethylene).
