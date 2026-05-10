# Neural Evasion: Real-Time Adversarial Evasion via Graph Neural Networks

**A complete pipeline** — from headless physics simulation through supervised GNN training to a real-time interactive game running at 60 fps on a commodity CPU with no GPU or cloud dependency.

An AI-controlled jet continuously evades a user-fired multi-missile barrage using a 198,872-parameter Graph Neural Network. The model encodes every active missile as an isolated interaction sub-graph, routes per-missile embeddings through a GRU-based cross-threat attention module that preserves trajectory memory across frames, and outputs both a collision danger score and a six-class tactical maneuver recommendation — all in ~4 ms on an Intel Core i5.

> **Report:** *Real-Time Adversarial Evasion via Graph Neural Networks with Temporal Threat Attention and Curriculum Learning* See [`paper/report.pdf`](paper/report.pdf).

---

## Repository Structure

```
neural-evasion/
│
├── main/
│   └── main3.py                    # V3 game — final system (play this one)
│
├── colab/
│   └── architecture.ipynb          # Full training notebook — 5-stage curriculum
│
├── simulation/
│   └── headless_sim3.py            # V3 data generator — 5-stage curriculum sim
│
├── paper/
│   └── report.pdf                  # IEEE conference paper
│
├── requirements.txt
└── README.md

# Legacy versions (see Version History below)
# v1.0 — main.py + headless_sim2.py
# v2.0 — main2.py + headless_sim2.py
```

---

## Quickstart — Play the Game

```bash
# 1. Install dependencies (read requirements.txt for PyG installation order)
pip install pygame torch torch_geometric numpy

# 2. Download trained weights  (jet_brain_v3.pt) from the v3.0 release assets
#    and place in the same directory as main3.py

# 3. Run
cd main
python main3.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `H` | Fire homing missile (tracks the jet) |
| `B` | Fire ballistic missile (fires at jet's current position — time your shot) |
| `C` | Fire cluster swarm — 5 micro-missiles in a V-spread (6 s cooldown) |
| `ESC` | Return to menu |

**Objective:** Take down the AI jet before the 15-second timer expires. If it survives, it counter-strikes your launchers.

> The game runs without `jet_brain_v3.pt` — it will warn and use random weights, which still demonstrates the threading and FSM systems.

---

## How the AI Works

The GNN runs in a background daemon thread, completely decoupled from the 60 fps physics loop via a pair of `maxsize=1` queues. This is a hard real-time guarantee — the render thread never waits for the model.

**Every ~100 ms, the GNN thread:**
1. Samples 12 candidate heading deltas
2. Projects the scene state forward 30 frames kinematically for each candidate
3. Builds one isolated sub-graph per active missile per candidate (the per-missile decomposition prevents a distant slow missile from diluting the GNN signal about a nearby fast one)
4. Runs a single **mega-batched** ThreatGNN forward pass over all sub-graphs simultaneously — one CPU call regardless of missile count
5. Routes per-missile embeddings through a **GRU-based attention** module that preserves trajectory memory across ticks
6. Picks the heading with the lowest danger score
7. Returns attention weights to the HUD (the live bars in the top-right corner show which missile the model is currently focusing on)

When danger crosses geometric thresholds, the model also recommends one of six FSM maneuvers: **NORMAL, BARREL_ROLL, JINKING, FALLING_LEAF, COBRA, IMMELMANN** — each with distinct flare deployment patterns.

---

## Training Your Own Weights

### Step 1 — Generate curriculum data (run locally)

```bash
cd simulation
python headless_sim3.py
```

Generates five JSONL files (one per curriculum stage):
- `data_slow_singles.jsonl` — 1–2 missiles, speed 2–4, low hot-spawn probability
- `data_fast_homing.jsonl` — 1–3 missiles, speed 4–6, tighter turn rate
- `data_multi_threat.jsonl` — 2–4 missiles
- `data_cluster_swarm.jsonl` — 3–6 missiles
- `data_adversarial.jsonl` — 4–8 missiles, speed 5.5–9, 60% close-spawn

Each simulation runs 300 frames. Labels are assigned retroactively: `label=1` if any missile-jet collision occurs within the next 60 frames (one full second at 60 fps). Maneuver labels are assigned by a geometric oracle that classifies each frame into one of six FSM states purely from scene geometry. Total: ~330,000 training records across all five stages.

### Step 2 — Train on Google Colab T4

Upload all five JSONL files to Colab, then open and run `colab/architecture.ipynb` top to bottom.

The notebook:
- Builds per-missile PyTorch Geometric `Data` objects (sub-graphs) with 5-dim context vectors `[norm_dist, tti, closing_norm, aoa, mtype]`
- Trains `NeuralEvasionBrain` (ThreatGNN + TemporalThreatAttention + ManeuverHead) with two-loss multi-task training: `L = BCE(danger, y) + 0.4 × CE(maneuver_logits, m_label)`
- Advances to the next curriculum stage when validation AUC plateaus (patience=3, max 8 epochs per stage)
- Saves best weights to `jet_brain_v3.pt`
- Produces an attention visualization: each missile is plotted with size and line opacity proportional to its attention weight, with the GNN's maneuver recommendation displayed per scene

Download `jet_brain_v3.pt` and place it next to `main3.py`.

> **Expected training time:** ~45–60 minutes for all five stages on a Colab T4 GPU. Each stage processes between 40k–95k graph records. A `.pt` cache file is written after each stage so interrupted sessions can resume without rebuilding graphs.

---

## Results (50-Trial Adversarial Benchmark)

| Metric | Baseline (V2) | Proposed (V3) | Δ |
|--------|:---:|:---:|:---:|
| Survival rate | 34.0% | **84.0%** | +50.0 pp |
| Avg. frames survived | 159 / 300 | 265 / 300 | +106 |
| Median inference (ms) | 45.6 | **20.7** | −24.9 ms |
| Val AUC | 0.9281 | **0.9384** | +0.0103 |
| Parameters | 174,914 | 198,872 | +23,958 |

The V3 system is both more accurate **and** faster than V2, despite having 24k more parameters. The latency drop comes from the mega-batch ThreatGNN design: all K missiles share a single batched forward pass rather than K sequential calls.

Single-tick latency by missile count (12,342 iterations, CPU only):

| K missiles | Avg (ms) | P95 (ms) |
|:---:|:---:|:---:|
| 1 | 3.86 | 5.76 |
| 2 | 4.02 | 5.88 |
| 3 | 4.14 | 5.92 |
| 4 | 4.27 | 6.28 |

P95 of 5.97 ms fits comfortably within a single 16.7 ms frame budget.

---

## Version History

### V1.0 — `main.py` + `headless_sim2.py` — *Legacy baseline*

The initial deployment. Introduces the core per-missile sub-graph decomposition and cross-threat attention architecture.

- **Model:** `ThreatGNN` + `CrossThreatAttention` (MLP-based, no temporal memory)
- **Context vector:** 2-dim `[norm_dist, tti]` per missile
- **Attention:** learned softmax over MLP scores, no GRU
- **Candidates:** 9 heading deltas searched per tick
- **Data:** single-stage, random-walk jet, 8,000 simulations
- **Weights:** `jet_brain_v1.pt` (174,914 parameters)
- **Survival rate:** 34% over 50 trials
- **Median latency:** 45.6 ms

### V2.0 — `main2.py` + `headless_sim2.py` — *Game polish, same architecture*

Same GNN architecture as V1, but significant game-side improvements: 5 scripted maneuvers added to the FSM (COBRA, FALLING_LEAF, BARREL_ROLL, JINKING, IMMELMANN), cluster swarm weapon, live attention weight HUD bars, neon radar visual aesthetic, and start/game-over screens with animated buttons.

- **Model:** identical to V1 — `ThreatGNN` + `CrossThreatAttention`, 2-dim context
- **Changes:** game UI, all 5 maneuvers, cluster weapon, visual effects
- **Weights:** `jet_brain_v2.pt`

### V3.0 — `main3.py` + `headless_sim3.py` + `architecture.ipynb` — *Final system*

Full architectural upgrade. Replaces the stateless attention module with a GRU-based temporal attention that maintains per-missile trajectory memory across inference frames. Adds a six-class maneuver head trained via multi-task learning. Expands the data pipeline to a five-stage curriculum.

- **Model:** `ThreatGNN` + `TemporalThreatAttention` (GRU, hidden=64) + `ManeuverHead`
- **Context vector:** 5-dim `[norm_dist, tti, closing_norm, aoa, mtype]`
- **Candidates:** 12 heading deltas (expanded action space)
- **Training:** 5-stage curriculum, 330,134 total frames, oracle-supervised maneuver labels
- **Weights:** `jet_brain_v3.pt` (198,872 parameters)
- **Survival rate:** 84% over 50 trials
- **Median latency:** 20.7 ms

---

## Notes on Infrastructure and Reproducibility

**Compute:** Data generation runs on any CPU in ~15–20 minutes per stage. Training requires a CUDA GPU — a free Colab T4 is sufficient. Inference at game-time runs on CPU with no GPU required.

**Threading:** The game uses Python's `threading` module with two `maxsize=1` `queue.Queue` objects — `state_queue` (game → GNN) and `decision_queue` (GNN → game). The maxsize=1 constraint acts as a natural rate limiter and prevents queue buildup. GNN inference latency spikes never affect the render thread.

**Data versioning:** Three distinct data generations correspond to the three versions. V1/V2 share `headless_sim2.py` (random-walk jet, single stage, no curriculum). V3 uses `headless_sim3.py` (50/50 flee-or-random jet behavior, 5 curriculum stages with escalating missile speed and count, per-stage turn-rate alignment between sim and game engine). The corrected oracle (margin=45 px, down from the initial 110 px bug) is embedded directly in both `headless_sim3.py` and in the notebook's `build_record()` function, which re-computes maneuver labels on the fly and ignores any stale `maneuver` field in the JSONL.

**Model packaging:** Weights are saved with `torch.save(model.state_dict(), 'jet_brain_v3.pt')`. Loading uses shape-filtered strict=False to handle minor architecture mismatches across experiments gracefully.

---

## Known Limitations

- **FSM–GNN gap:** The GNN maneuver head recommends FALLING_LEAF (≥2 missiles within 150 px) more often than the FSM executes it — the geometry gate is stricter than the model's learned intuition. End-to-end policy gradient training is the natural fix.
- **2D only:** The simulator and game engine are 2D. Missile homing uses proportional navigation in the screen plane only.
- **GRU resets on missile count change:** When a missile is destroyed, the GRU hidden state resets for all remaining missiles, losing accumulated trajectory memory at the moment it is most critical.

---

## Citation

```bibtex
@inproceedings{bukhari2025neuralevasion,
  title     = {Real-Time Adversarial Evasion via Graph Neural Networks with Temporal Threat Attention and Curriculum Learning},
  author    = {Bukhari, Syed Hamza Mukhtar},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2025},
  institution = {GIK Institute of Engineering Sciences and Technology}
}
```
