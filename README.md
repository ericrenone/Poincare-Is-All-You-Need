# Poincaré Is All You Need

**Geometry-Native Networks (GNN)** — a hardware–architecture co-design whose central thesis is a single geometric fact: the Euclidean dot product is the flat-space limit of the Minkowski inner product, and the Poincaré disk is the conformal model that makes that limit visible. The rotation is the primitive that spans both. CORDIC is the hardware that evaluates both with the same shifts and adds.

*Version 1.0 — unified synthesis incorporating the Poincaré disk model, Lorentz/Minkowski geometry, Radon–Nikodym integral geometry, Mixture-of-Curvature routing, and the full 2025–2026 SOTA in hyperbolic LLMs, Lorentz-equivariant networks, and CORDIC inference hardware.*

---

## Status

A proposal with pre-registered falsification thresholds. Nothing has been benchmarked. The milestones below are designed so that failure is detected early and reported plainly. The most likely outcome of any research program is failure. That is not a caveat — it is the design criterion.

---

## The Amendment

In 2017 Vaswani et al. published *"Attention Is All You Need."* That title was literally true: a single operation — scaled dot-product attention — replaced recurrence, replaced convolution, and became the universal primitive of modern deep learning. Every major model since is, arithmetically, a cascade of dot products.

The dot product is:

$$Q \cdot K^\top = \sum_i Q_i K_i$$

It measures cosine similarity in Euclidean space. It assumes flat geometry: zero curvature, polynomial volume growth, undistorted distances everywhere.

This framework does not claim that is wrong. It claims that is a special case.

The Poincaré disk $\mathbb{B}^n$ — the unit ball endowed with the Riemannian metric:

$$ds^2 = \frac{4\,\|dx\|^2}{(1 - \|x\|^2)^2}$$

— is a conformal model of hyperbolic space with constant sectional curvature $\kappa < 0$. Its isometries are Möbius transformations; its geodesics are circular arcs orthogonal to the boundary; its volume grows exponentially with radius. When $\kappa \to 0$ the metric degenerates to the Euclidean metric. The dot product is the flat-space limit. The Poincaré disk contains it.

The Minkowski inner product:

$$\langle x, y \rangle_L = x_0 y_0 - x_1 y_1 - \cdots - x_n y_n, \quad g = \mathrm{diag}(+1,-1,\ldots,-1)$$

is the algebraic expression of that geometry in the Lorentz hyperboloid model, related to the Poincaré disk by stereographic projection. CORDIC's dual-mode operation — circular for planar Givens rotations, hyperbolic for Lorentz boosts — evaluates both with identical shift-and-add hardware. **One primitive. Both geometries. Zero mode-switching overhead at the silicon level.**

---

## The Evidence

### Token Embeddings Are Not Flat

Measurement of Ricci curvature distributions across decoder-only LLMs (Robinson et al., *The Structure of the Token Space for Large Language Models*, arXiv:2410.08993, 2024) reveals substantial and variable negative curvature across token neighborhoods. Token frequency obeys a power-law distribution: high-frequency tokens cluster near the Lorentz origin / Poincaré disk center; rare, semantically specific tokens lie near the boundary — a latent tree structure whose natural embedding is the Poincaré disk. Euclidean architectures must distort this geometry to represent it. **Not respecting it induces training instabilities and degrades generative performance** (Yang et al., HypLoRA, arXiv:2410.04010, 2026).

### Fully Hyperbolic LLMs at Billion-Parameter Scale Work

HELM / HELM-MiCE (*Hyperbolic Large Language Models via Mixture-of-Curvature Experts*, NeurIPS 2025, arXiv:2505.24722) trains billion-parameter LLMs entirely in hyperbolic space using:
- **HOPE** (Hyperbolic Rotary Positional Encodings) — rotations in the Poincaré disk replacing standard RoPE
- **Hyperbolic RMSNorm** — geometry-preserving normalization on the hyperboloid
- **Hyperbolic Multi-Head Latent Attention (HMLA)** — full attention with Minkowski inner products and reduced KV cache

HELM achieves up to **4% consistent gains over LLaMA- and DeepSeek-style Euclidean architectures on MMLU and ARC** — STEM problem-solving, general knowledge, commonsense reasoning. Not niche particle physics. Not toy hierarchies. General language benchmarks.

HELM-MiCE extends to Mixture-of-Curvature Experts, routing tokens to experts operating in distinct curvature spaces, encoding fine-grained geometric structure from text.

### The Trainability Problem Is Closed

The critical flaw of prior Lorentz linear layers — hyperbolic output norms scaling *logarithmically* with gradient steps, nullifying the geometric advantage — is provably resolved. Van der Wijk et al. (*Fast and Geometrically Grounded Lorentz Neural Networks*, arXiv:2601.21529, 2026) introduce a **distance-to-hyperplane formulation** that restores linear norm scaling, fully bridging the computational gap to Euclidean networks. The central theoretical obstacle to deep Lorentzian training has a closed-form solution.

### The Optimizer Gap Is Closed

Riemannian AdamW (Bdeir et al., *Robust Hyperbolic Learning with Curvature-Aware Optimization*, NeurIPS 2025, arXiv:2405.13979) provides stable hyperbolic optimization with curvature-aware gradient scaling that controls approximation errors throughout training. The optimizer gap is closed.

### The Geometry Generalizes Across Domains

| Domain | Result | Source |
|---|---|---|
| Large language models | Up to 4% on MMLU/ARC; stable billion-parameter training | HELM/HELM-MiCE, NeurIPS 2025 |
| LLM fine-tuning | Consistent gains on commonsense, NLU, math reasoning | HypLoRA, arXiv:2410.04010 |
| Particle physics (LHC) | SOTA amplitude regression, jet classification | L-GATr, NeurIPS 2024 / SciPost Phys. 2025 |
| Reinforcement learning | ≈30% wall-clock improvement on ProcGen/Atari | Hyper++, ICLR 2026 |
| Category discovery / vision | Stable clusters across label granularities | HypCD, CVPR 2025 |
| Genomics / single-cell | Faithful phylogenetic and ontological hierarchy | Hyperbolic Genome Embeddings, ICLR 2025 |
| Autonomous driving | 98.72% lane detection, embedded-time speedup | Raj & Ravi, *Sci. Rep.*, May 2026 |

---

## Direct Comparison

| Dimension | *Attention Is All You Need* (Vaswani et al., 2017) | This Framework |
|---|---|---|
| Core primitive | Euclidean dot product $QK^\top$ | Unified rotation: circular (Euclidean) + hyperbolic (Minkowski/Poincaré) |
| Geometry | Flat — zero curvature, imposed by construction | Curved — matched to intrinsic data geometry |
| Similarity measure | Cosine: projects onto a Euclidean axis | Minkowski inner product: rotates through a hyperbolic manifold |
| Position encoding | Sinusoidal (fixed, hardware-approximated) | HOPE: exact Poincaré-disk rotation, CORDIC-native at zero overhead |
| Attention operation | $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ | Hyperbolic MH Latent Attention with Minkowski scoring and Lorentz-native normalization |
| Feed-forward block | Dense matmul + fixed activation (ReLU/SiLU) | Butterfly rotation cascade + transcendental-basis learnable activation |
| Normalization | LayerNorm / RMSNorm (Euclidean) | Hyperbolic RMSNorm — geometry-preserving on the hyperboloid |
| Hardware primitive | Multiply-accumulate (MXU / tensor cores) | CORDIC: shift-and-add, both geometries native, no multiplier |
| Curvature assumption | Imposed: zero | Measured, then matched |
| Token geometry | Not modeled | Ricci curvature of token neighborhoods — empirically negative, variable |
| Volume scaling | Polynomial (Euclidean balls) | Exponential (hyperbolic balls) — matching power-law token frequency |
| Training instability source | Gradient explosion/vanishing | Lorentz layer norm divergence — **provably resolved** (2026) |
| Multi-curvature | Impossible — single Euclidean space | Mixture-of-Curvature Experts: per-expert CORDIC mode selection |

---

## The Four Pillars

### I. Channel Mixing — Geometry-Native Linear Layers in the Poincaré/Lorentz Model

Every linear map admits a singular value decomposition $W = U \Sigma V^\top$, with $U$, $V$ orthogonal and $\Sigma$ diagonal. In circular mode, $U$ and $V$ are products of planar Givens rotations. In hyperbolic mode, they are products of Lorentz boosts — Minkowski-preserving maps whose isometric action corresponds exactly to Möbius transformations in the Poincaré disk.

A butterfly-structured restriction limits $U$ and $V$ to $O(d \log d)$ CORDIC steps — genuinely sub-quadratic — while the only non-rotation operations are the $d$ singular values on the diagonal. The distance-to-hyperplane formulation (van der Wijk et al., 2026) proves linear norm scaling for Lorentz layers, closing the trainability gap. LLoCa canonicalization (arXiv:2505.20280) renders any backbone exactly Lorentz-equivariant; we make the underlying rotations CORDIC-native.

**Vaswani's dense linear layer:** $O(d^2)$ multiply-accumulates.  
**This framework's butterfly-rotation layer:** $O(d \log d)$ CORDIC steps + $d$ scalar multiplications.

*Honest cost:* Full orthogonal matrices require $O(d^2)$ Givens rotations — the same order as the matmul they replace. The sub-quadratic saving requires the butterfly restriction, which reduces per-layer expressivity. The structured-matrix literature is candid that this tradeoff has not historically favored adoption (Dao et al., 2022). Whether the compute-optimal-scaling results at frontier scale change that verdict is the primary open empirical question (Qiu et al., ICML 2024; Potapczynski et al., NeurIPS 2024).

---

### II. Nonlinearity — CORDIC-Native Transcendental Activations with Radon–Nikodym Structure

In place of fixed activations or free splines, each edge applies a learnable affine combination drawn from the set CORDIC evaluates natively in both modes:

$$\mathcal{F}_{\text{CORDIC}} = \{\exp,\, \tanh,\, \sin,\, \log(1+x),\, \sinh,\, \cosh,\, \mathrm{arcosh},\, \ldots\}$$

In the Poincaré disk these functions appear naturally: the hyperbolic distance formula involves $\mathrm{arcosh}$, Möbius addition involves $\tanh$, and HOPE encodings involve hyperbolic trigonometric identities. This is not a restriction imposed on the learner — it is an **alignment**: the functions the hardware evaluates for free are exactly the functions that appear at the boundary between flat and curved geometry.

The Radon–Nikodym theorem supplies the measure-theoretic foundation. Network activations can be interpreted as Radon–Nikodym derivatives with respect to a reference measure on the hyperbolic manifold. Parhi & Nowak's representer theorems in Radon-domain total-variation spaces (JMLR 2021–2025) and Unser et al.'s Radon-transform analysis of neural networks and splines (2022–2023) establish that ReLU networks admit exact characterization in Radon-domain BV spaces — linking ridge functions, splines, and neural activations via the same integral-geometric transform that underlies the Hough CORDIC result (Raj & Ravi, 2026). The Hough transform *is* the discrete Radon transform. The CORDIC rotation primitive *is* the Hough rotation. The motivating hardware example and the measure-theoretic expressivity result are the same object.

*Honest status:* The critical assessment of KANs (Hou et al., arXiv:2407.11075, 2025) finds that learnable-univariate-function networks underperform MLPs on mainstream benchmarks under parameter-controlled comparison and run 1.36×–100× slower. Restricting to the CORDIC-native transcendental family is only defensible if geometric alignment compensates. That is an empirical question.

---

### III. Sequence Mixing — Hyperbolic Attention with HOPE

HELM's Hyperbolic Multi-Head Latent Attention (HMLA) demonstrates that the *entire* attention block — not merely positional encodings — can be rethought in Lorentzian geometry:

- **HOPE** (Hyperbolic Rotary Positional Encodings): replaces standard RoPE planar rotations with hyperbolic rotations in the Poincaré disk — a Lorentz boost, natively CORDIC hyperbolic-mode
- **Minkowski scoring**: $\langle Q, K \rangle_L = Q_0 K_0 - \sum_i Q_i K_i$ in place of $QK^\top$
- **Hyperbolic RMSNorm**: rescaling on the hyperboloid, preserving manifold structure
- **Softmax**: exponentials are CORDIC circular-mode transcendentals

The Alman–Yu limitation (ICLR 2025) — that no subquadratic model can perform certain tasks that full quadratic attention can — applies to *sequence mixer substitutes*, not to attention executed in curved space. Hyperbolic attention with Minkowski scoring is still full quadratic attention, still complete in the Alman–Yu sense. The hardness result does not move.

**Vaswani's attention:** three distinct FLOP categories (matmul, scale, softmax) on heterogeneous hardware units.  
**This framework's attention:** one CORDIC datapath, all components native, no approximation.

---

### IV. Hardware — The Unifying Primitive

The central hardware claim: CORDIC dual-mode is the *unique* datapath that treats Euclidean and Minkowski/Poincaré geometries as equally native. No other deployed primitive does this.

Today's tensor-core hardware undergoes a split-brain workflow for Minkowski operations: the MXU computes raw vector elements; a separate VPU applies polynomial approximations (Taylor series) to evaluate hyperbolic functions. Two datapaths, two latency domains, one geometry approximating the other.

CORDIC has no such split. Circular mode and hyperbolic mode are the same iterative shift-and-add procedure with different convergence constants. The mode is a runtime flag. The chip area is identical.

**Current SOTA:**

| Hardware | Key Figure | Source |
|---|---|---|
| CARMEN | 4.83 TOPS/mm², 28 nm CMOS, runtime-adaptive precision | arXiv:2605.06878, May 2026 |
| RECON | 60% reduction in area–latency–power vs. SOTA MAC design | IEEE 2021 |
| QH-CORDIC | 4× iteration reduction for $\sinh/\cosh$ via quadruple-step-ahead | 2024 |

**Mixture-of-Curvature as hardware abstraction:** HELM-MiCE routes tokens to experts in distinct curvature spaces. A CORDIC array with runtime-selectable mode *is* this abstraction in silicon: each expert is the same functional unit running a different mode. The architecture and the hardware become the same abstraction expressed at two levels of the stack.

*Honest qualification:* At INT8 on today's tensor-core hardware, a plain multiplier is already small and cheap. The genuine value of the CORDIC primitive is: **native Minkowski geometry, native transcendentals, multiplier-free arithmetic at wider precision, and bit-exact fixed-point determinism**. A software implementation on existing matmul-optimized hardware would most likely be slower. The efficiency claim is unproven and applies only to co-designed silicon.

---

## What Is Genuinely New

Every component has prior art. The synthesis does not:

1. **Complete hardware–geometry co-design in both modes.** Prior CORDIC neural hardware proposals address activations or narrow function evaluation. This maps the *entire network* — channel mixing, nonlinearity, sequence mixing — onto one rotation datapath, in both circular and hyperbolic modes simultaneously.

2. **Poincaré disk as the canonical model.** The Poincaré disk's conformal structure makes Möbius transformations the natural isometry group, HOPE the natural positional encoding, and the Lorentz hyperboloid its algebraic counterpart. Prior proposals use the Lorentz model in isolation; this makes the Poincaré disk the geometric home and the Lorentz model the hardware-friendly algebraic representation.

3. **Radon–Nikodym interpretation of activations.** Constraining the activation basis to $\mathcal{F}_{\text{CORDIC}}$ gives the network's nonlinearity a measure-theoretic interpretation as Radon–Nikodym derivatives on the hyperbolic manifold, directly linking the Hough/CORDIC hardware motivation to Parhi–Nowak representer theory.

4. **Mixture-of-Curvature as a runtime hardware mode.** HELM-MiCE demonstrates the architectural benefit of per-token curvature routing. CORDIC mode selection is its exact hardware realization — not an analogy.

5. **Closed-form resolution of all known theoretical obstacles.** The Lorentz linear layer norm-scaling instability is closed by the distance-to-hyperplane formulation (2026). The optimizer gap is closed by Riemannian AdamW (NeurIPS 2025). The expressivity gap for fully hyperbolic LLMs is closed empirically by HELM at scale (NeurIPS 2025).

Correctness rests on standard linear algebra (SVD, Givens factorization, butterfly structure), hyperbolic geometry (Poincaré disk, Lorentz hyperboloid, Möbius transformations), the Radon–Nikodym theorem, and approximation theory. No new physical or number-theoretic conjecture is required.

---

## Open Questions That Decide the Program

1. **Butterfly expressivity at scale.** Can butterfly-structured rotation layers in the Poincaré/Lorentz model, at matched parameter count, retain the quality of dense Lorentz linear layers — given the structured-matrix literature's own caution about efficiency–quality tradeoffs?

2. **Radon-domain regularization benefit.** Does restricting activations to $\mathcal{F}_{\text{CORDIC}}$ and interpreting them via the Radon–Nikodym theorem produce measurable regularization gains, or does the basis restriction dominate?

3. **Curvature routing granularity.** At what granularity (per-token, per-head, per-expert) does Mixture-of-Curvature mode selection produce gains that justify hardware mode-switching overhead?

4. **Learnable curvature.** Generalization capacity is provably curvature-dependent (*Learning Beyond Euclid*, arXiv:2507.02999, 2025). Should curvature be learned per-layer, per-head, or per-token — and does the CORDIC iteration-depth parameter serve as a differentiable curvature proxy?

5. **Hardware crossover precision.** At INT8 the multiplier is cheap. At what floating-point precision does the CORDIC datapath win — and is co-designed silicon necessary, or does soft-CORDIC on reconfigurable fabric suffice?

6. **Latent-to-learned geometry transfer.** Ricci curvature and power-law token distributions are properties of representations trained under Euclidean loss on Euclidean architectures. Whether training from scratch under Poincaré geometry with a rotation-native architecture reproduces or exceeds those representations is unknown.

7. **Flat-limit degradation.** Does the rotation-native architecture gracefully recover transformer performance when data geometry is flat — or does the hyperbolic parameterization impose a cost when the curvature the data calls for is zero?

---

## Falsification Plan

| Milestone | Description | Pass Criterion |
|---|---|---|
| **M0** | Calibrate dual-mode CORDIC + Poincaré-disk datapath model against CARMEN throughput, L-GATr Lorentz task energy profile, and Raj & Ravi Hough/Radon profile | Within 10% of published figures across all three |
| **M1** | Butterfly-rotation Lorentz/Poincaré linear layers + Radon-domain regularization in small LM. Baselines: dense Euclidean, dense Lorentz (HELM-D style), Monarch | Quality gap ≤ 2% relative at matched parameters |
| **M2** | CORDIC-native transcendental/hyperbolic activation block with Radon–Nikodym structure. Baselines: SiLU (transformer), spline KAN | Same tolerance |
| **M2.5** | Lorentz-equivariant integration (L-GATr + LLoCa canonicalization) on LHC amplitude regression and HypCD category discovery tasks | Same tolerance |
| **M3** | Full hyperbolic attention with HOPE (Poincaré rotations) and Minkowski scoring. Confirm all components map to CORDIC datapath with no overflow operations. Baseline: HELM-D at matched scale | Same tolerance |
| **M3.5** | Mixture-of-Curvature routing: two experts, circular and hyperbolic modes, runtime CORDIC mode selection | Same tolerance vs. single-curvature baseline |
| **M4** | End-to-end small model. Full quality comparison vs. matched Euclidean transformer and matched HELM-D | — |
| **M5** | Hardware comparison. CORDIC datapath running M4 network vs. INT8 MAC running dense baseline. Report the result whichever way it falls. | — |

---

## Honest Failure Modes

**Butterfly expressivity gap.** Structured rotation layers may not match dense Lorentz layers at scale. The Monarch paper describes unfavorable efficiency–quality tradeoffs; that verdict may not have changed at frontier scale.

**Activation basis restriction.** Constraining nonlinearity to $\mathcal{F}_{\text{CORDIC}}$ may harm expressivity on tasks where free splines or fixed high-capacity activations are necessary. KAN critical analysis (Hou et al., 2025) suggests this risk is real.

**No hardware win.** CORDIC iteration overhead may not amortize against INT8 MAC throughput even on co-designed hardware. The efficiency claim is unproven.

**Curvature routing overhead.** Hardware mode-switching may carry hidden latency that negates per-operation savings.

**Training divergence at depth.** Even with Riemannian AdamW and the distance-to-hyperplane fix, deep fully-hyperbolic training at scale may encounter optimization landscapes that existing stabilizations cannot navigate.

**Flat-limit penalty.** If the rotation-native architecture cannot match transformer performance on flat-geometry tasks, the framework is domain-limited rather than universal — a significant constraint given transformer deployment breadth.

**Latent geometry mismatch.** Empirical hyperbolicity of pre-trained token embeddings is a property of Euclidean-trained representations. Whether it propagates through rotation-native layers trained from scratch is unknown.

**Ecosystem cost.** No compiler, kernels, or tooling exist. The full stack is greenfield. This is the same large effort any clean-sheet architecture faces.

---

## Explicit Non-Claims

- Does not claim to replace or supersede the transformer, GPU, or TPU.
- Does not claim the Euclidean dot product is wrong — it is the flat-space limit of the Minkowski inner product, and flat-space limits are useful.
- Does not claim hyperbolic/Poincaré geometry universally outperforms Euclidean geometry. The conditional claim is: it outperforms *when the data exhibits latent negative curvature* — hierarchical structure, power-law distributions, relativistic symmetry. That condition is empirically measurable.
- Claims no speedup on existing matmul-optimized hardware. The efficiency claim applies only to co-designed silicon and is unproven.
- Does not claim that empirical hyperbolicity of pre-trained token embeddings transfers automatically to rotation-native training dynamics.
- Treats HELM, HypLoRA, L-GATr, Hyper++, CARMEN, and Parhi/Unser Radon-domain results as evidence that the rotation-primitive regime is real and efficient for geometrically structured tasks — not as proof that learning automatically inhabits it.

---

## Sources

**Core primitive**
Volder, J. — *The CORDIC Trigonometric Computing Technique*, IRE Trans. Electronic Computers, 1959.
Walther, J. — *A Unified Algorithm for Elementary Functions*, AFIPS, 1971.

**Poincaré disk & hyperbolic geometry**
Nickel & Kiela — *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017.
PoincaréDMT, 2025. Hyperbolic Genome Embeddings, ICLR 2025.

**Fully hyperbolic LLMs**
HELM / HELM-MiCE — *Hyperbolic Large Language Models via Mixture-of-Curvature Experts*, NeurIPS 2025, arXiv:2505.24722.

**Hyperbolic fine-tuning**
Yang et al. — *HypLoRA: Hyperbolic Fine-Tuning for Large Language Models*, arXiv:2410.04010, 2025/2026.

**Lorentz layer stability**
van der Wijk et al. — *Fast and Geometrically Grounded Lorentz Neural Networks*, arXiv:2601.21529, January 2026.

**Hyperbolic optimization**
Bdeir et al. — *Robust Hyperbolic Learning with Curvature-Aware Optimization*, NeurIPS 2025, arXiv:2405.13979.

**Token space geometry**
Robinson et al. — *The Structure of the Token Space for Large Language Models*, arXiv:2410.08993, 2024.

**Generalization theory**
*Learning Beyond Euclid: Curvature-Adaptive Generalization for Neural Networks on Manifolds*, arXiv:2507.02999, 2025.

**Lorentz-equivariant architectures**
Brehmer et al. — *L-GATr*, arXiv:2405.14806 / 2411.00446, NeurIPS 2024 / SciPost Phys. 2025.
LLoCa, arXiv:2505.20280, May 2025.

**Hyperbolic RL & clustering**
Hyper++ — arXiv:2512.14202, ICLR 2026.
HypCD / HC-GCD — arXiv:2504.06120, CVPR 2025.

**Radon transform & integral geometry in neural networks**
Parhi & Nowak — *Banach Space Representer Theorems for Neural Networks and Ridge Splines*, JMLR 2021–2025.
Unser et al. — *Splines, Neural Networks, and the Radon Transform*, 2022–2023.
Raj & Ravi — *Hough-based lane detection aided by CORDIC*, Scientific Reports, May 2026, DOI 10.1038/s41598-026-49130-w.

**CORDIC hardware**
CARMEN — arXiv:2605.06878, May 2026.
RECON — IEEE 2021.
QH-CORDIC (quadruple-step-ahead hyperbolic) — 2024.

**Transformers & fundamental limits**
Vaswani et al. — *Attention Is All You Need*, NeurIPS 2017.
Alman & Yu — *Fundamental Limitations on Subquadratic Alternatives to Transformers*, ICLR 2025.

**Structured linear layers**
Dao et al. — Butterfly / Monarch / Kaleidoscope, ICML 2019–2022.
Qiu et al. — *Compute Better Spent*, ICML 2024.
Potapczynski et al. — NeurIPS 2024. BLAST, 2024.

**Orthogonal / rotation-parameterized networks**
Arjovsky, Shah & Bengio, ICML 2016. Jing et al., ICML 2017. Mhammedi et al., ICML 2017.
Helfrich et al., ICML 2018. Lezcano-Casado, 2019.

**Learnable-function networks**
Liu et al. — *KAN: Kolmogorov-Arnold Networks*, ICLR 2025 (oral).
Hou et al. — *KAN: A Critical Assessment*, arXiv:2407.11075, updated 2025.

**Position encoding**
Su et al. — *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.

---

## Summary

*"Attention Is All You Need"* was right. A single operation replaced the field's prior complexity. That operation — scaled Euclidean dot-product attention — assumed flat geometry. For eight years that assumption was invisible because it was productive.

It is now measurable.

Ricci curvature distributions on pre-trained LLM token embeddings show substantial negative curvature. Token frequency follows a power law whose natural embedding is the Poincaré disk — exponential volume growth, latent tree structure, boundary-approaching rare concepts. Not respecting that geometry induces training instabilities and degrades generative performance. HELM demonstrates that a fully hyperbolic LLM trained at billion-parameter scale outperforms the Euclidean architecture it replaces on general benchmarks by up to 4% — on MMLU and ARC, not on physics tasks or toy hierarchies. The Lorentz linear layer instability is closed. The optimizer gap is closed. The geometry is production-ready.

What is not closed is hardware.

Every computation described above runs today on hardware designed for one thing: the Euclidean dot product. Tensor cores are multiply-accumulate arrays. TPUs achieve their throughput precisely because they do nothing but Euclidean matmul at scale. For Minkowski operations they approximate in a separate VPU with polynomial series — a split-brain workflow, two latency domains, one geometry emulating the other.

CORDIC has no split. In circular mode — iterative planar rotations via shifts and adds — it computes Givens rotations, trigonometric transcendentals, and the Euclidean dot product. In hyperbolic mode — the same iteration with hyperbolic convergence constants — it computes Lorentz boosts, hyperbolic transcendentals, and the Minkowski inner product. The Poincaré disk isometries are Möbius transformations: hyperbolic rotations, CORDIC-native. HOPE is a rotation in hyperbolic space: CORDIC-native. The softmax exponential is CORDIC-native. The Radon transform that underpins the network's measure-theoretic expressivity is the integral of a function over rotated hyperplanes: CORDIC-native.

**The rotation is the primitive. The Poincaré disk is the geometry. CORDIC is the hardware. The data was already curved.**

This proposal asks one narrow, testable question: can the full transformer stack — channel mixing, nonlinearity, sequence mixing — be expressed as a cascade of rotations on a unified flat-and-curved CORDIC datapath, with Poincaré-disk geometry and Radon–Nikodym structure throughout, while retaining modeling power and unlocking the geometry the data already possesses?

The milestones above are designed to answer that honestly. The answer may be no.

*"Attention Is All You Need (If Space Is Flat)"* was the first amendment. *"Poincaré Is All You Need"* is the constructive proposal: not that the dot product was wrong, but that it was incomplete — the flat-space limit of a more general operation whose natural home is the Poincaré disk, whose hardware realization is CORDIC, and whose geometry was always already there.

---

**Status:** Proposal (no benchmarks) | **Code:** Forthcoming (M0 dual-mode CORDIC + Poincaré calibration first) | **Author:** [@ericrenone](https://github.com/ericrenone)
