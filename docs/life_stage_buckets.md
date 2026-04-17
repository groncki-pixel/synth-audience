---
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
---

# Life Stage Buckets

**Defaults and assumptions used in this document**

- **RNG seed:** 42  
- **Embedding dimension:** 128  
- **Platform universe:** tiktok; instagram; facebook; youtube; discord; reddit; whatsapp; snapchat; twitter  
- **Age cutoffs:** Child 0–10; Adolescent 11–15; Young Adult 16–22; Young Professional 23–34; Adult 35–64; Older Adult 65+  

All defaults are explicit so they can be replaced by empirical priors later.

---

### Child

**Age range**

0–10

**Behavioral profile**

Children have low cognitive autonomy and high dependence on caregivers for media access and interpretation. Their household role is primarily receiver and learner; siblings and parents shape tastes and exposures. Primary social contexts are family routines, early childcare, and supervised playgroups. Dominant platforms and channels are family streaming services, YouTube Kids, educational apps, and linear TV under parental control.

**Typical daily media pathways**

- Morning: family TV or supervised tablet time → educational app  
- Afternoon: caregiver‑mediated YouTube Kids or streaming with siblings  
- Evening: family streaming or bedtime audio stories

**Decision making style**

Parental mediated heuristic — choices are made by caregivers; children follow simple cues (bright visuals, repetition, parental approval).

**Default platform exposure windows (hours per day distribution)**

- **YouTube Kids / family streaming:** 0.5–2.0 hours/day (mean 1.0)  
- **Educational apps:** 0.25–1.0 hours/day (mean 0.5)  
- **Passive TV/audio:** 0.25–1.0 hours/day (mean 0.5)

**Why this bucket matters to Hollywood stakeholders**

Children drive early franchise affinity and long‑term IP value; family viewing patterns determine early word‑of‑mouth and parental purchase decisions for toys, licensing, and family marketing.

---

### Adolescent

**Age range**
**Typical daily media pathways**
- Morning: short social feeds during commute or before school  
- Afternoon: school peer sharing, Discord or group chat interactions  
- Evening: TikTok/YouTube bingeing and sharing with friends

**Decision making style**

Status and identity seeking — exploratory but socially calibrated; decisions weigh peer approval and trend visibility.

**Default platform exposure windows (hours per day distribution)**

- **TikTok / short video:** 1.0–3.0 hours/day (mean 1.8)  
- **YouTube:** 0.5–2.0 hours/day (mean 1.0)  
- **Discord / group chat:** 0.25–1.5 hours/day (mean 0.6)

**Why this bucket matters to Hollywood stakeholders**

Adolescents are trend accelerants; they create and normalize memes and fandom behaviors that can make or break youth‑oriented releases and social campaigns.

---

### Young Adult

**Age range**

16–22

**Behavioral profile**

Young adults show high exploratory autonomy and rapid peer mixing across online communities. They are often students or early career entrants with fluid social networks spanning campus, online fandoms, and creator communities. Platforms include TikTok, Instagram, Reddit, and Discord; creators and memes act as cross‑cluster bridges.

**Typical daily media pathways**

- Morning: social feeds and news snippets  
- Daytime: community forums, creator content, short videos between tasks  
- Evening: long‑form streaming, social viewing parties, subreddit threads

**Decision making style**

Exploratory and social proof driven — open to novelty but sensitive to peer cluster signals and creator endorsements.

**Default platform exposure windows (hours per day distribution)**

- **TikTok / Instagram:** 1.0–2.5 hours/day (mean 1.6)  
- **Reddit / Discord:** 0.5–1.5 hours/day (mean 0.8)  
- **Long‑form streaming:** 0.5–2.0 hours/day (mean 1.0)

**Why this bucket matters to Hollywood stakeholders**

Young adults are early adopters and cultural translators; they convert niche creator trends into mainstream attention and are key to festival, indie, and streaming breakout success.

---

### Young Professional

**Age range**

23–34

**Behavioral profile**

Young professionals balance career formation with active social lives; they have high media autonomy but constrained time budgets. Social contexts include workplaces, friend groups, and professional networks. Platforms skew toward Instagram, TikTok, Twitter/X, and LinkedIn for professional signals; private messaging apps mediate close‑network sharing.

**Typical daily media pathways**

- Morning: quick social check and news headlines  
- Workday: short breaks for feeds and creator content  
- Evening: curated streaming, social sharing with friend groups

**Decision making style**

Instrumental and status aware — choices optimize social capital and personal identity; receptive to high‑quality, time‑efficient content.

**Default platform exposure windows (hours per day distribution)**

- **Instagram / TikTok:** 0.75–2.0 hours/day (mean 1.2)  
- **Twitter/X / LinkedIn:** 0.25–0.75 hours/day (mean 0.4)  
- **Streaming:** 1.0–2.5 hours/day (mean 1.4)

**Why this bucket matters to Hollywood stakeholders**

Young professionals are high‑value consumers for premium subscriptions, theatrical attendance, and targeted advertising; they influence household purchase decisions and social prestige signaling.

---

### Adult

**Age range**

35–64

**Behavioral profile**

Adults have stable preferences and moderate media autonomy; household and workplace responsibilities shape exposure. Social contexts include family, workplace, and community groups. Platforms emphasize Facebook, YouTube, WhatsApp, and news sites; influencer impact is lower but trusted sources and groups matter.

**Typical daily media pathways**

- Morning: news and curated feeds  
- Daytime: work‑related browsing and short video breaks  
- Evening: family streaming, long‑form video, group chats

**Decision making style**

Pragmatic and trust oriented — decisions rely on trusted sources, family needs, and established tastes.

**Default platform exposure windows (hours per day distribution)**

- **Facebook / YouTube:** 0.75–2.0 hours/day (mean 1.2)  
- **WhatsApp / group messaging:** 0.25–1.0 hours/day (mean 0.6)  
- **Streaming / long‑form:** 1.0–3.0 hours/day (mean 1.6)

**Why this bucket matters to Hollywood stakeholders**

Adults represent the largest share of box office and subscription revenue; their established tastes and household purchasing power determine long‑tail revenue and merchandising success.

---

### Older Adult

**Age range**

65+

**Behavioral profile**

Older adults often have lower platform adoption but high loyalty to preferred channels; household roles include grandparents and community anchors. Primary social contexts are family, local community groups, and legacy media consumption. Platforms include Facebook, YouTube, and messaging apps; linear TV remains important.

**Typical daily media pathways**

- Morning: linear news or curated social posts  
- Afternoon: community group interactions and video content on YouTube  
- Evening: family streaming or TV with family members

**Decision making style**

Conservative and authority oriented — decisions favor trusted institutions, familiar genres, and recommendations from close social ties.

**Default platform exposure windows (hours per day distribution)**

- **Facebook / YouTube / TV:** 1.0–3.0 hours/day (mean 1.8)  
- **Messaging apps:** 0.25–1.0 hours/day (mean 0.5)

**Why this bucket matters to Hollywood stakeholders**

Older adults provide stable, predictable revenue and are influential in family viewing decisions; they are critical for legacy IP monetization and cross‑generational marketing.

---

### Cross‑bucket notes and implementation guidance

- **Null and latent semantics** — children and early adolescents will have many **latent** fields (ideology_score, full big5 vectors) that become explicit in later buckets; document these transitions in trait rules.  
- **Platform exposure windows** are **distributions** not fixed values; sampler should draw per‑agent from truncated normal or beta distributions parameterized by the means above.  
- **Provenance** — every generated example agent must include `metadata.source_weights`, `metadata.created_by`, and `metadata.seed` to ensure reproducibility.

---

### Short overall pitch for Hollywood stakeholders

Life‑stage buckets convert demographic labels into behavioral personas that predict how content spreads, who amplifies it, and which channels convert attention into revenue. By modeling **who** sees, **how** they react, and **through whom** trends travel at each life stage, studios and marketers can design targeted release strategies, optimize creator partnerships, and forecast box office and streaming demand with greater fidelity than demographic‑only approaches.

---


Adolescents are in a transitional phase with increasing autonomy but still strongly shaped by household norms. They occupy school and peer group contexts where identity and status begin to form. Dominant platforms are TikTok, Snapchat, YouTube, and Discord; peer clusters and school networks amplify trends rapidly.
11–15
