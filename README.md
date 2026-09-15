CGMas (Coarse-Graining Multi-Agent System)

Joohee Choi<sup>a†</sup>, Junhyeong Lee<sup>b†</sup>, and Seunghwa Ryu<sup>a,b,c*</sup>

<sup>a</sup> Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Republic of Korea <sup>b</sup> KAIST InnoCORE PRISM-AI Center, Korea Advanced Institute of Science and Technology (KAIST), Republic of Korea <sup>c</sup> Department of AX, Korea Advanced Institute of Science and Technology (KAIST), Republic of Korea

<sup>†</sup> These authors contributed equally. <sup>*</sup> Corresponding author: ryush@kaist.ac.kr

Code Overview

Bottom-up coarse-grained (CG) molecular dynamics extends polymer simulation beyond the scales accessible to all-atom (AA) methods, but deriving a CG force field is laborious because the mapping is user-defined and potentials must be rebuilt for every polymer and resolution. Existing tools take the atomistic topology and the bead mapping as given, so the stages where errors originate remain with the user. We present CGMas (Coarse-Graining Multi-Agent System), a multi-agent framework that automates topology construction, equilibration, bead mapping, Boltzmann-inversion potential derivation, and validation directly from a natural-language description of the polymer and target resolution. Rather than relying on one-shot generation, CGMas pairs an LLM reasoning layer with deterministic physics-aware tools and bounded self-correction loops: a reasoning agent builds the OPLS-AA constitutional repeat unit, four validators return the specific violated constraint together with its local bonding context, and independent review agents screen the assembled data file and the fitted potentials before the CG model is run (Figure 1). The framework supports homopolymers and copolymers across random, alternating, block, and periodic architectures. CGMas represents a step toward physics-based, AI-driven automation of molecular-simulation pipelines.

<img width="7480" height="4410" alt="fig1" src="https://github.com/user-attachments/assets/4049ef40-d7d5-433d-b962-388e3a513b94" />

Overview of CGMas for automation of polymer coarse-graining

We assessed the robustness of CGMas using a twenty-seven-task benchmark organized as a five-level difficulty ramp: hydrocarbons (4 tasks), heteroatom-substituted repeat units (7), complex side chains (9), multifunctional repeat units (3), and copolymers (4). CGMas completed all 27 tasks, matched the AA equilibrium density to within 5% in 22 of them, and reduced the effective simulation turnaround from 38–88 min to 1 min per task (Figure 2). Reasoning cost was approximately $0.002 per task. The framework was further verified to be reproducible across repeated runs, insensitive to the construction and simulation parameters, and robust to the choice of LLM backbone.

<img width="5727" height="5186" alt="fig2" src="https://github.com/user-attachments/assets/c8d071d7-df61-4b0a-8487-ce5f4b962712" />
