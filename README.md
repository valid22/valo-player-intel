# VALORANT Player Intelligence

## Abstract

This project studies whether real VALORANT player behavior can be segmented into interpretable archetypes, whether those archetypes differ between professional and public play, and whether pre-match behavioral structure carries predictive signal for match outcomes.

Using real public data from organized VCT play and public competitive matches, the system builds a canonical player-match dataset, engineers behavioral features, applies unsupervised clustering at both the global and role-conditioned levels, and trains calibrated win-probability models. A separate agent-behavior layer tests whether agents are actually used according to their nominal class or drift into different behavioral roles in practice.

The current evidence suggests three main conclusions:

- role-conditioned clustering is substantially more interpretable than one global cross-role clustering pass
- pre-match player-history and archetype-composition features carry strong predictive signal for match outcomes
- agent usage is mostly aligned with nominal role in pro play, but public play shows materially more behavioral drift

## Research Questions

This work is organized around six questions:

1. Do coherent behavioral clusters exist in real VALORANT match data?
2. Are those clusters more interpretable when conditioning on role?
3. Do player archetypes differ between pro and public cohorts?
4. Do archetype-composition features add value to pre-match win modeling?
5. Can agents be grouped by actual usage behavior rather than official class labels?
6. When agents are clustered by behavior, do they align with their nominal role or drift elsewhere?

## Current Dataset

Current corpus size:

- professional / organized cohort:
  - matches: `224`
  - player-match rows: `2235`
- public competitive cohort:
  - matches: `568`
  - player-match rows: `5792`

Data sources:

- organized play: public VLR-backed event and match endpoints
- public competitive play: Henrik public VALORANT API

The repository does not ship synthetic match logs.

## Headline Results

### Unsupervised structure

Global cohort clustering:

- pro global silhouette: `0.2663`
- public global silhouette: `0.2835`

Role-conditioned clustering:

- pro Duelist silhouette: `0.2940`
- pro Controller silhouette: `0.2585`
- pro Initiator silhouette: `0.2864`
- pro Sentinel silhouette: `0.2791`
- public Duelist silhouette: `0.2267`
- public Controller silhouette: `0.2611`
- public Initiator silhouette: `0.3088`
- public Sentinel silhouette: `0.2413`

Interpretation:

- global clustering yields useful exploratory structure but still mixes overlapping cross-role behavior
- role-conditioned clustering is the more credible lens for “real” VALORANT archetypes
- the strongest role-specific separation in the current public sample appears in Initiators

### Supervised outcome modeling

Best current model by cohort:

- pro:
  - model: `hist_gradient_boosting`
  - Brier: `0.0009`
  - ROC AUC: `1.0000`
  - F1: `1.0000`
  - Brier improvement vs baseline: `84.7%`
- public:
  - model: `hist_gradient_boosting`
  - Brier: `0.0161`
  - ROC AUC: `0.9974`
  - F1: `0.9737`
  - Brier improvement vs baseline: `89.8%`

Interpretation:

- pre-match player-history features are strongly informative
- team archetype composition features are useful enough to retain in the supervised pipeline
- calibrated offline models materially outperform the static strength-gap baseline

### Agent-behavior alignment

Agent behavior is inferred from unsupervised clustering over agent-level behavioral profiles rather than from Riot’s nominal role labels.

Current alignment rates:

- pro:
  - represented agents: `27`
  - raw alignment rate: `59.3%`
  - stable alignment rate: `100.0%`
  - low-sample agents: `8`
- public:
  - represented agents: `27`
  - raw alignment rate: `70.4%`
  - stable alignment rate: `78.9%`
  - low-sample agents: `0`

Interpretation:

- once low-sample agents are separated from the stable set, pro alignment is very strong
- public play shows more agent-role drift, which is consistent with looser coordination and broader usage patterns
- low-sample agents are now retained and explicitly marked as insufficient evidence rather than silently dropped

## Method

### 1. Canonicalization

Raw source pulls are normalized into a common schema with:

- match metadata
- player identifiers
- team identifiers
- map
- outcome
- combat statistics
- objective interaction
- agent identity

The normalization step also:

- parses mixed datetimes
- standardizes agent names
- removes invalid agent strings

### 2. Behavioral feature engineering

Per-player features include:

- win rate
- KDA ratio
- kills / deaths / assists per match
- headshot rate
- damage proxy
- entry rate and entry success rate
- support score
- objective score
- consistency score
- role entropy
- role concentration
- map pool entropy

### 3. Unsupervised analysis

Three unsupervised layers are used:

- global cohort clustering
- role-conditioned clustering
- agent-behavior clustering

Global clustering is intended for broad behavioral structure.
Role-conditioned clustering is intended for interpretable VALORANT archetypes.
Agent-behavior clustering is intended to test nominal-vs-actual usage.

### 4. Supervised analysis

Match-level modeling includes:

- static baseline
- logistic regression
- histogram gradient boosting
- PyTorch MLP benchmark

Evaluation includes:

- Brier score
- log loss
- ROC AUC
- average precision
- accuracy
- balanced accuracy
- precision
- recall
- F1
- expected calibration error

### 5. Composition features

The match model now includes team-level composition summaries derived from the unsupervised layer:

- global archetype counts
- role-archetype counts
- archetype diversity
- archetype balance
- opponent-gap versions of the same features

## Main Findings

### Finding 1

The original “one clustering pass over everyone” approach was analytically weak for VALORANT-specific interpretation.

The current evidence supports using:

- global clustering for exploratory structure
- role-conditioned clustering for interpretable VALORANT archetypes

### Finding 2

Professional and public cohorts differ not only in outcome signal but in behavioral coherence.

The public cohort remains noisier, but still contains usable structure, especially once role conditioning is applied.

### Finding 3

Outcome modeling is stronger than unsupervised separation alone.

This matters because it means:

- archetypes are informative
- but continuous behavioral features and matchup composition still carry additional signal beyond discrete cluster membership

### Finding 4

Agent behavior is not perfectly equivalent to official role labels.

Some agents align very cleanly with their nominal class.
Others drift, especially in public play.
This is a meaningful result rather than noise: it quantifies how players actually use agents.

### Finding 5

Low-sample agents are analytically dangerous.

The current pipeline now keeps them visible but separates them from stable alignment claims instead of silently dropping them or overcommitting to weak inference.

### Finding 6

Archetype prevalence is now traceable over time.

The timeline layer is descriptive rather than causal, but it is already useful for showing how cluster participation changes across the observed window. It is currently strongest for public data and partially available for pro data because pro timestamps are now present for `197 / 224` matches.

## Interpretation

The project is now useful in three ways.

1. As a behavioral analytics system:
   it discovers and compares player styles across pro and public environments.

2. As a predictive modeling system:
   it shows that player-history and archetype-composition features are strongly informative for pre-match outcome estimation.

3. As an agent-usage system:
   it tests whether agents are used according to design intent or repurposed behaviorally by the player base.

The second and third points are where most of the practical value currently sits.

## Limitations

- global clusters are still broad and should not be mistaken for definitive role archetypes
- pro timestamps are now partially recovered from VLR detail payloads, but timeline coverage is still stronger for public than pro
- low-sample agent behavior should be treated as descriptive only
- supervised metrics are based on offline holdout evaluation, not deployment-grade live validation
- cluster-to-outcome summaries are associative, not causal

## Deliverables

Generated outputs include:

- `data/interim/matches.parquet`
- `data/interim/player_matches.parquet`
- `data/processed/player_features.parquet`
- `data/processed/match_level.parquet`
- `artifacts/segmentation/*.parquet`
- `artifacts/prediction/model_metrics.json`
- `artifacts/prediction/model_predictions.parquet`
- `artifacts/prediction/calibration_curves.parquet`
- `reports/figures/*.html`
- `results.json`

The Streamlit report includes:

- Overview
- Segmentation
  - Global
  - Role-Specific
  - Agent Behavior
- Modeling
- Comparison
- Data

## Reproducibility

Install:

```bash
.venv/bin/python -m pip install -e .[dev]
```

Fetch real data:

```bash
HENRIK_API_KEY=... PYTHONPATH=src .venv/bin/python -m valo_player_intel.cli fetch
```

Run the full pipeline:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m valo_player_intel.cli run --manifest data/external/source_manifest.json
```

Run the app:

```bash
PYTHONPATH=src .venv/bin/streamlit run src/valo_player_intel/app/streamlit_app.py
```

Run tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Repository Structure

- `data/raw/`: raw fetched source files
- `data/interim/`: normalized canonical tables
- `data/processed/`: feature-engineered tables
- `src/valo_player_intel/`: ingestion, feature engineering, clustering, prediction, reporting, app
- `artifacts/segmentation/`: clustering, role-specific, and agent-behavior outputs
- `artifacts/prediction/`: model metrics, predictions, calibration curves
- `reports/figures/`: saved Plotly HTML figures
- `tests/`: unit tests

