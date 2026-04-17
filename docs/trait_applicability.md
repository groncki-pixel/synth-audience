---
title: "Trait Applicability and Inheritance Rules"
rng_seed: 42
embedding_dim: 128
platform_universe: ["tiktok","instagram","facebook","youtube","discord","reddit","whatsapp","snapchat","twitter"]
age_cutoffs:
  child: "0-10"
  adolescent: "11-15"
  young_adult: "16-22"
  young_professional: "23-34"
  adult: "35-64"
  older_adult: "65+"
created_by: "TG"
output: md_document
---

# Trait Applicability and Inheritance Rules

**Purpose**
This document codifies which agent traits are **explicit**, **latent**, **inherited**, or **emergent** by life stage, and provides the exact mathematical inheritance and peer‑mixing functions, sampling pseudocode, and implementation notes needed to instantiate agents deterministically from priors and donor corpora.

**Scope**
Covers the trait set required by `schemas/life_stage_agent.schema.json` and the life‑stage buckets defined in `docs/life_stage_buckets.md`. All parameter defaults are explicit and replaceable by empirical priors.

---

## 1. Trait inventory and canonical definitions

**Modeled traits (canonical names)**

- **ideology_score** — continuous $[0,1]$; political / ideological leaning; may be `null` for latent cases.
- **big5** — object with keys `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`; each $[0,1]$ or `null`.
- **moral_foundations** — object with keys `care`, `fairness`, `loyalty`, `authority`, `sanctity`; each $[0,1]$ or `null`.
- **taste_vector** — fixed‑length embedding (dimension = 128 by default) or `null`.
- **media_autonomy** — number $[0,1]$ probability agent chooses media independently.
- **peer_influence_weight** — number $[0,1]$ weight of peer signals in decisions.
- **parental_influence_weight** — number $[0,1]$ weight of parental/household signals.
- **virality_susceptibility** — number $[0,1]$ composite derived from peer_influence_weight and platform_usage.
- **platform_usage** — object mapping platform names to intensity $[0,1]$.
- **memory_store** — short text entries (LLM conditioning).
- **exposure_history** — list of exposures (content_id, timestamp, source, reaction).
- **decision_rules** — object describing thresholds and parameters for watch/share decisions.
- **metadata** — created_at, source_weights, seed, created_by.

**Trait statuses**
- **explicit** — sampled or assigned directly from priors/donors and stored.
- **latent** — not directly observable; may be `null` until later life stages; can be imputed when needed.
- **inherited** — sampled from household parent(s) via inheritance function.
- **emergent** — produced by peer mixing or life‑stage maturation rules.

---

## 2. Trait applicability matrix (life stage → trait status)

| Trait | Child (0–10) | Adolescent (11–15) | Young Adult (16–22) | Young Professional (23–34) | Adult (35–64) | Older Adult (65+) |
|---|---:|---:|---:|---:|---:|---:|
| **ideology_score** | **inherited** (household) | **weak** (inherited + noise) | **explicit** | **explicit** | **explicit** | **explicit** |
| **big5** | **latent** (null) | **partial** (extraversion, openness emerging) | **explicit** | **explicit** | **explicit** | **explicit** |
| **moral_foundations** | **latent** | **partial** | **explicit** | **explicit** | **explicit** | **explicit** |
| **taste_vector** | **inherited** | **mixed** (household + peer) | **explicit** | **explicit** | **explicit** | **explicit** |
| **media_autonomy** | **inherited/low** | **emergent/medium** | **explicit/high** | **explicit/high** | **explicit/medium-high** | **explicit/low-medium** |
| **peer_influence_weight** | **0** | **high** | **high** | **moderate** | **low** | **low** |
| **parental_influence_weight** | **high** | **moderate** | **low** | **low** | **low** | **low** |
| **platform_usage** | **inherited/parental** | **emergent** | **explicit** | **explicit** | **explicit** | **explicit** |
| **virality_susceptibility** | **derived** | **derived** | **derived** | **derived** | **derived** | **derived** |

**Notes**
- "Partial" means only a subset of subdimensions are sampled explicitly (e.g., extraversion, openness) while others remain latent.
- "Derived" traits are computed from other traits and platform usage (see Section 5).

---

## 3. Household inheritance functions

**Purpose**
For children (0–10) and early adolescents (11–13), certain traits are inherited from parent agent(s) in the same `household_id`. The inheritance function is a weighted average with additive noise.

**Notation**
- Let $T_{child}$ be the child trait vector (scalar or vector).
- Let $T_{parent}$ be the parent trait vector (if multiple parents, use weighted average of parents).
- Let $\alpha$ be the inheritance weight (stage dependent).
- Let $\varepsilon$ be additive noise sampled from $\mathcal{N}(0,\sigma^2)$ applied elementwise and then truncated to $[0,1]$.

**Default parameters**
- Children (0–10): $\alpha = 0.9$
- Early adolescents (11–13): $\alpha = 0.6$
- Noise: $\varepsilon \sim \mathcal{N}(0, 0.05^2)$ (apply per trait component; truncate to $[-0.2,0.2]$ before adding)

**Formula (scalar trait)**

$$T_{child} = \alpha \cdot T_{parent} + (1-\alpha)\cdot \varepsilon$$

**Formula (vector trait, e.g., taste_vector)**

$$\mathbf{T}_{child} = \alpha \cdot \mathbf{T}_{parent\_avg} + (1-\alpha)\cdot \boldsymbol{\varepsilon}$$

where 
$\mathbf{T}_{parent\text{avg}} = \frac{\sum_{p \in parents} w_p \mathbf{T}_p}{\sum_p w_p}$ 
and $w_p$ are parent weights (default equal unless household metadata indicates primary caregiver).

**Implementation notes**
- After computing, clamp each component to $[0,1]$.
- For categorical or enum traits (e.g., education_level), sample from parent distribution with small mutation probability $p_{mut}=0.02$.

---

## 4. Peer influence mixing (adolescents and young adults)

**Purpose**
Adolescents (11–15) and young adults (16–22) blend their taste vectors with peer cluster centroids to model socialization and trend adoption.

**Notation**
- $\beta$ — peer mixing coefficient (stage dependent).
- $\mathbf{T}_{agent}$ — agent taste vector after household inheritance (if any).
- $\mathbf{C}_{peer}$ — centroid of sampled peer cluster (vector).
- $\boldsymbol{\eta}$ — small noise vector $\mathcal{N}(0,\sigma^2)$.

**Default parameters**
- Adolescents: $\beta = 0.4$
- Young adults: $\beta = 0.25$
- Noise $\sigma = 0.03$

**Formula**

$$\mathbf{T}_{new} = (1-\beta)\cdot \mathbf{T}_{agent} + \beta \cdot \mathbf{C}_{peer} + \boldsymbol{\eta}$$

**Peer cluster sampling**
- Peer clusters are sampled from the population graph using platform overlap and school/workplace membership.
- For synthetic sampling when no graph exists: sample $k$ donors from donor embeddings conditioned on age_bucket and platform_usage, compute centroid.

**Implementation notes**
- Normalize vectors (unit length) if embeddings are cosine‑based; otherwise keep raw and clamp to allowed range.
- Record `metadata.source_weights` showing fraction from household vs peer donors.

---

## 5. Trait evolution across life stages

**Principles**
- Traits move from **latent → partial → explicit** as agents age.
- Transition ages and rates are configurable; defaults below.

**Default transitions**
- **big5**: latent in child; extraversion and openness begin emerging at 11–15; full explicit by 16.
- **moral_foundations**: latent in child; partial by adolescence; explicit by young adult.
- **ideology_score**: inherited until 16; explicit sampling allowed at 16+ with household prior as prior distribution.

**Mathematical smoothing for maturation**
When converting a latent trait to explicit, sample from a prior centered on inherited value with maturation noise that decays with age:

$$T_{explicit} \sim \mathcal{N}\big(T_{inherited}, \sigma_{mature}^2 \cdot e^{-\lambda (age - age_{onset})}\big)$$

Default: $\sigma_{mature}=0.15$, $\lambda=0.2$, $age_{onset}$ = first age where trait becomes explicit.

---

## 6. Platform usage and donor mapping

**Platform usage assignment**
- For each agent, sample `platform_usage[platform]` from age‑conditional priors (if available) or from default truncated normal distributions parameterized in `docs/life_stage_buckets.md`.
- Normalize platform usage so that the maximum is ≤ 1; do not force sum to 1 (these are intensities).

**Donor mapping for taste_vector**
- If agent has high media_autonomy and explicit taste_vector, map to nearest donor embedding using cosine similarity on demographic proxies (age_bucket, gender, coarse geography) and embedding distance.
- Record `metadata.source_weights` as `{ "household": x, "peer_cluster": y, "donor_match": z }`.

**Fallback**
- If no donor embeddings available, sample taste_vector from a mixture of household vector and a small random vector drawn from a global prior.

---

## 7. Virality susceptibility (derived trait)

**Definition**
Virality susceptibility is a composite score combining peer influence weight, platform usage intensity on high‑virality platforms, and personality signals (e.g., extraversion).

**Formula (example)**

$$S_{virality} = \sigma\Big( w_p \cdot peer\_influence\_weight + w_{pl} \cdot \overline{platform\_virality} + w_e \cdot extraversion \Big)$$

where:
- $\sigma$ is a logistic squashing function to map to $[0,1]$.
- $\overline{platform\_virality} = \sum_{i \in P_{high}} platform\_usage_i / |P_{high}|$ (average usage on high‑virality platforms; default $P_{high}=\{\text{tiktok, instagram, reddit}\}$).
- Default weights: $w_p=0.5, w_{pl}=0.35, w_e=0.15$.

**Implementation**
- Compute `virality_susceptibility` at agent instantiation and update after major life events (peer cluster change, platform adoption).

---

## 8. Sampling pseudocode

```text
function instantiate_agent(household, age, life_stage, priors, donors, rng):
    agent = {}
    agent.agent_id = uuid4(rng)
    agent.age = age
    agent.life_stage = life_stage
    agent.household_id = household.id if household else null

    # 1. Platform usage
    agent.platform_usage = sample_platform_usage(age, priors.platform_usage_by_age, rng)

    # 2. Parental/household inheritance for applicable traits
    if life_stage in ['child','adolescent_early']:
        parents = household.parent_agents
        parent_taste = weighted_average([p.taste_vector for p in parents])
        alpha = 0.9 if life_stage == 'child' else 0.6
        agent.taste_vector = clamp(alpha * parent_taste + (1-alpha) * normal_vector(0,0.05), 0,1)
        agent.media_autonomy = clamp(alpha * mean([p.media_autonomy for p in parents]) + noise(), 0,1)
        agent.ideology_score = clamp(alpha * mean([p.ideology_score for p in parents]) + noise(), 0,1)
    else:
        # explicit sampling or donor mapping
        if priors.has_donors and agent.media_autonomy > 0.6:
            agent.taste_vector, source_weight = donor_map(agent, donors)
            agent.metadata.source_weights = source_weight
        else:
            agent.taste_vector = sample_from_global_taste_prior(rng)

    # 3. Peer mixing for adolescents and young adults
    if life_stage in ['adolescent','young_adult']:
        peer_cluster = sample_peer_cluster(agent, household, priors, rng)
        beta = 0.4 if life_stage == 'adolescent' else 0.25
        agent.taste_vector = normalize((1-beta)*agent.taste_vector + beta*peer_cluster.centroid + normal_vector(0,0.03))

    # 4. Big5 and moral foundations
    if life_stage == 'child':
        agent.big5 = {k: null for k in big5_keys}
        agent.moral_foundations = {k: null for k in mf_keys}
    elif life_stage == 'adolescent':
        agent.big5 = sample_partial_big5(priors, rng)  # extraversion, openness explicit
        agent.moral_foundations = sample_partial_mf(priors, rng)
    else:
        agent.big5 = sample_big5(priors, agent.demographics, rng)
        agent.moral_foundations = sample_mf(priors, agent.demographics, rng)

    # 5. Derived traits
    agent.virality_susceptibility = compute_virality_susceptibility(agent)

    # 6. Decision rules defaults
    agent.decision_rules = default_decision_rules_for_stage(life_stage)

    # 7. Metadata
    agent.metadata = {
        "created_at": now_utc(),
        "created_by": "TG",
        "seed": rng.current_seed(),
        "source_weights": agent.metadata.source_weights if exists else {}
    }

    return agent
```

**Notes**
- All sampling functions must accept an RNG object for deterministic reproducibility.
- `clamp` enforces $[0,1]$ bounds.
- `normalize` applies vector normalization if embeddings are cosine‑based.

---

## 9. Decision rule schema (summary)

`decision_rules` object fields:

- **watch_threshold** — numeric baseline threshold in $[0,1]$. Function of virality_susceptibility, global_hype, and neighbor_influence.
- **share_threshold** — numeric threshold in $[0,1]$. Function of reaction_strength, social_capital, and exposure_count.
- **reconsideration_window** — integer days for re‑evaluation (default 7).
- **decay_rate** — daily decay of interest (default 0.12).
- **k_complex** — number of distinct neighbor exposures required for complex contagion (bucket dependent).

**Example functional forms**

Watch threshold:

$$watch\_threshold = \max\Big(0,\ \tau_0 - \alpha_s \cdot S_{virality} - \alpha_h \cdot global\_hype - \alpha_n \cdot neighbor\_influence\Big)$$

Share threshold:

$$share\_threshold = \sigma\Big(\gamma_0 + \gamma_r \cdot reaction\_strength + \gamma_s \cdot social\_capital - \gamma_e \cdot exposure\_fatigue\Big)$$

Default coefficients: $\tau_0=0.6$, $\alpha_s=0.35$, $\alpha_h=0.25$, $\alpha_n=0.4$. $\gamma_0=-1.0$, $\gamma_r=1.2$, $\gamma_s=0.8$, $\gamma_e=0.5$.

**Bucket complex contagion k defaults**
- Adolescents: $k=2$ (require exposures from 2 distinct neighbors)
- Young adults: $k=2$
- Young professionals: $k=1$ or $k=2$ depending on content type
- Adults / Older adults: $k=1$

---

## 10. Pseudocode: exposure → decision → memory update

```text
function process_exposure(agent, content, timestamp, source):
    # 1. record exposure
    agent.exposure_history.append({content_id: content.id, timestamp: timestamp, source: source})

    # 2. compute neighbor influence
    neighbor_influence = compute_neighbor_influence(agent, content)

    # 3. compute watch probability
    watch_threshold = agent.decision_rules.watch_threshold
    watch_score = f_watch(agent.virality_susceptibility, content.global_hype, neighbor_influence)
    will_watch = (watch_score >= watch_threshold)

    if will_watch:
        reaction_strength = generate_reaction(agent, content)  # LLM or heuristic
        # 4. compute share probability
        share_threshold = agent.decision_rules.share_threshold
        share_score = f_share(reaction_strength, agent.social_capital, agent.exposure_history)
        will_share = (share_score >= share_threshold)

        if will_share:
            emit_share(agent, content, timestamp)

        # 5. update memory_store
        agent.memory_store = update_memory(agent.memory_store, content, reaction_strength, timestamp)

    # 6. apply decay to interest
    decay_agent_interest(agent, timestamp)
```

**LLM conditioning for generate_reaction**
Include: short agent persona (age_bucket, top 3 platform usages, taste_vector summary), recent memory_store entries (last 3), content metadata (title, genre, short excerpt), and platform context. Keep prompt under token budget and redact PII.

---

## 11. Implementation and engineering notes

- **Order of sampling**: platform_usage → household inheritance → donor mapping → peer mixing → personality/moral sampling → derived traits → decision_rules → metadata. This order ensures dependencies are available when needed.
- **Determinism**: pass a seeded RNG object through all sampling functions; store seed in metadata.seed.
- **Missing data**: allow `null` for latent traits; provide imputation functions that use household priors or global priors when required for simulation.
- **Provenance**: every trait sampled from donors must include `metadata.source_weights` with keys `donor_id`, `weight`, `confidence`.
- **Validation**: include unit tests that check trait ranges, inheritance weight behavior (e.g., child trait moves toward parent mean), and peer mixing effects (e.g., cosine similarity increases after mixing).
- **Privacy**: donor IDs must be hashed and PII removed before storage; record license and provenance in `data/data_license.md`.

---

## 12. Parameter table (defaults)

| Parameter | Default value | Notes |
|---|---|---|
| child inheritance α | 0.9 | high household influence |
| early adolescent α | 0.6 | partial household influence |
| adolescent β (peer mix) | 0.4 | peer centroid mixing |
| young adult β | 0.25 | weaker peer mixing |
| inheritance noise σ | 0.05 | applied elementwise |
| maturation σ_mature | 0.15 | for latent→explicit sampling |
| maturation decay λ | 0.2 | age decay rate |
| virality weights (w_p, w_pl, w_e) | (0.5, 0.35, 0.15) | for S_virality |
| complex contagion k (adolescent) | 2 | exposures from distinct neighbors |
| complex contagion k (adult) | 1 | single exposure sufficient |

All parameters are configurable in `config/defaults.yaml` for experiments.

---

## 13. Acceptance tests (for this document)

- **Reproducibility**: given a fixed RNG seed and a small household with parent vectors, child instantiation yields identical taste_vector across runs.
- **Inheritance sanity**: child taste_vector cosine similarity to parent average > 0.85 with α=0.9 (unit tests should assert this within tolerance).
- **Peer mixing effect**: adolescent taste_vector moves toward peer centroid by approximately β fraction (test with synthetic centroid).
- **Range checks**: all numeric traits remain in $[0,1]$ after sampling and clamping.
- **Decision rules**: watch/share thresholds produce plausible watch/share rates on a small synthetic content set (smoke test).
