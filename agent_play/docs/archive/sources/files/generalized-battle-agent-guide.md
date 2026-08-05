# Training a Generalized Battle Agent for fheroes2

## Entity-centric state modeling, structured ability encoding, legal-action scoring, and simulator-guided learning

**Scope:** tactical battles in [fheroes2](https://github.com/ihhub/fheroes2), rather than adventure-map play  
**Repository state reviewed:** `master`, accessed August 5, 2026  
**Primary design goal:** one low-latency policy that can play many army compositions, not a separate policy for every fixed matchup

---

## 1. Problem formulation

A fixed battle such as five Peasants against five Peasants is a useful environment test, but it is a poor formulation of the final learning problem. It uses a degenerate initial-state distribution,

\[
s_0 \sim \rho_{\mathrm{fixed}}(s)
=
\delta(s-s_{\text{5 Peasants versus 5 Peasants}}).
\]

A generalized battle agent should instead be trained over a distribution of initial battles,

\[
s_0 \sim \rho_{\mathrm{train}}(s),
\]

where \(\rho_{\mathrm{train}}\) varies:

- creature types;
- stack counts;
- numbers of creatures per stack;
- army strength;
- formations;
- battlefield obstacles;
- attacker and defender orientation;
- stochastic battle seeds;
- eventually, hero spells and siege conditions.

The desired policy is therefore a universal conditional policy,

\[
\pi_\theta(a\mid s),
\]

or a state-action value model,

\[
Q_\theta(s,a),
\]

whose input contains the current armies and board state. The policy is not expected to memorize a lookup table for every matchup. It should learn transferable concepts such as reachability, kill thresholds, retaliation, focus fire, target value, protection of shooters, and turn-order interactions.

The practical objective is:

\[
\boxed{
\text{learn reusable tactical evaluation over a distribution of battles}
}
\]

rather than:

\[
\boxed{
\text{solve one initial battle configuration}
}
\]

### 1.1 Levels of generalization

It is useful to distinguish several claims that are often all called “generalization.”

| Level | Test condition | Main requirement |
|---|---|---|
| Seed generalization | Familiar armies with new random seeds or obstacles | Robustness within the training distribution |
| Composition generalization | New combinations of familiar creatures | Entity-centric and relational representation |
| Count generalization | Familiar creatures at unseen count or army-strength ranges | Sensible numerical normalization and threshold modeling |
| Creature generalization | A held-out creature whose component mechanics appeared in training | Semantic stat and ability representation |
| Ability-combination generalization | Familiar ability primitives combined in a new way | Compositional ability encoding |
| Mechanic generalization | A genuinely new operation never represented during training | Explicit rule semantics or simulator access |

The first four are realistic targets for a compact policy. The last one cannot be obtained merely by assigning a new integer ID to an unseen mechanic. A model must either receive the mechanic in terms of known semantic primitives or use the engine to evaluate its consequences.

---

## 2. The engine should remain the authority on rules

The fheroes2 engine already knows:

- which actions are legal;
- how movement and wide-unit geometry work;
- how attacks are resolved;
- which units are affected by area attacks;
- when retaliation occurs;
- how creature abilities and weaknesses interact;
- how stochastic damage and ability procs are sampled;
- how turn order and status effects change.

The learned component should not reproduce all of this from scratch. Its main job should be to compare legal tactical choices and estimate their long-term consequences.

A productive division of labor is:

\[
\boxed{
\text{engine: legality and exact transition rules}
}
\]

\[
\boxed{
\text{network: tactical evaluation and long-horizon approximation}
}
\]

This reduces invalid actions, improves sample efficiency, and makes generalization to unusual unit combinations easier.

The current fheroes2 battlefield has width 11 and height 9, for 99 cells. The engine exposes board distance, adjacency, reachability, wide-unit positioning, and attack-position checks in `battle_board.h`.[^board]

The battle loop calls the built-in planner from `Battle::Arena::UnitTurn` when the active unit is AI-controlled. This is a natural place to insert an external policy adapter.[^arena] The engine command representation already distinguishes movement, attack, spellcast, skip, retreat, surrender, and other command types. An attack command includes attacker, defender, movement cell, attack cell, and direction.[^command]

---

## 3. Abilities are structured data, not natural-language input

The textual descriptions visible to a player are only a presentation layer.

In the current repository, each creature ability is represented by:

```cpp
struct MonsterAbility
{
    MonsterAbilityType type;
    uint32_t percentage;
    uint32_t value;
};
```

Each creature’s battle statistics contain:

```cpp
std::vector<MonsterAbility> abilities;
std::vector<MonsterWeakness> weaknesses;
```

`MonsterAbilityType` is an enum containing entries such as:

```cpp
DOUBLE_SHOOTING
DOUBLE_MELEE_ATTACK
MAGIC_RESISTANCE
SPELL_CASTER
UNLIMITED_RETALIATION
NO_ENEMY_RETALIATION
HP_DRAIN
AREA_SHOT
ENEMY_HALVING
SOUL_EATER
```

The source code separately converts these records to phrases such as “Double shot” for the user interface.[^monster-info-header][^monster-info-source] The policy should read the structured records before that conversion. No language model, tokenizer, or text-to-text interface is required.

For example, the repository defines abilities in forms such as:

```cpp
MonsterAbilityType::DOUBLE_SHOOTING
MonsterAbilityType::MAGIC_RESISTANCE, 25, 0
MonsterAbilityType::SPELL_CASTER, 20, Spell::BLIND
```

The second form contains a 25 percent parameter. The third contains a 20 percent probability and a spell identifier. These are already machine-readable values.[^monster-assignments]

### 3.1 What not to do

Do not feed the raw enum integer as a continuous scalar:

\[
x_{\mathrm{ability}} = 17.
\]

The numerical order of enum values has no semantic meaning. A network should not infer that ability 18 is “closer” to ability 17 than ability 30 is.

Do not use the localized description string as the model input. Text introduces unnecessary tokenization cost, localization dependence, and ambiguity. It also obscures exact numerical fields such as proc probability and spell ID.

Do not rely only on creature ID. A creature-ID embedding can be included as a residual feature, but a policy trained primarily on IDs can memorize matchup-specific behavior and fail on held-out creatures or new combinations.

---

## 4. Recommended ability representation

The recommended design has three layers:

1. **Raw engine records** preserve the exact source-of-truth fields.
2. **Semantic ability tokens** describe the rule using reusable primitives.
3. **Action-conditioned effect summaries** describe what the rule does in the current state and for the current candidate action.

These layers serve different purposes.

### 4.1 Layer 1: raw engine record

Export every ability and weakness without converting it to text.

```json
{
  "type_id": 16,
  "percentage": 20,
  "value": 7
}
```

For logs, the observation may also contain names such as `"SPELL_CASTER"` and `"BLIND"`. These names should not be passed through a language model. They are converted to categorical IDs before inference.

A minimal neural encoding is:

\[
e_k
=
E_{\mathrm{type}}[t_k]
+
E_{\mathrm{payload}}[v_k]
+
W_n n_k,
\]

where:

- \(t_k\) is the ability type;
- \(v_k\) is a categorical payload such as spell ID;
- \(n_k\) contains normalized numerical fields such as probability or magnitude.

The ability embeddings for a creature can be pooled:

\[
e_{\mathrm{abilities}}
=
\rho\left(
\sum_{k=1}^{K}
\phi(e_k)
\right),
\]

where \(\phi\) and \(\rho\) are small multilayer perceptrons. This is a Deep Sets-style encoder. A small attention pool is also reasonable, but the number of abilities per creature is sufficiently small that simple sum pooling is a strong baseline.

Using a set of tokens is preferable to one fixed slot per ability type because a creature can have multiple records of the same general type with different payloads, such as immunity or damage reduction for several specific spells.

### 4.2 Layer 2: semantic ability token

The raw `type`, `percentage`, and `value` fields are compact, but `value` is type-dependent. For `SPELL_CASTER`, it is a spell ID. For another ability, it can be a magnitude or a different identifier. Add a deterministic adapter in C++ that maps each raw record to a typed semantic schema.

A useful schema is:

```cpp
enum class AbilityTrigger : uint8_t {
    ALWAYS,
    ON_MOVE,
    ON_MELEE_ATTACK,
    ON_RANGED_ATTACK,
    ON_ATTACK,
    ON_HIT,
    ON_DAMAGE_DEALT,
    ON_DAMAGE_RECEIVED,
    ON_INCOMING_SPELL,
    ON_RETALIATION,
    ON_TURN_START,
    ON_DEATH,
    BATTLE_START
};

enum class AbilityOperation : uint8_t {
    TAG,
    SET_FOOTPRINT,
    SET_MOVEMENT_MODE,
    REPEAT_ATTACK,
    MODIFY_DAMAGE,
    MODIFY_RETALIATION,
    MODIFY_MELEE_PENALTY,
    MODIFY_TARGET_PATTERN,
    CANCEL_EFFECT,
    SCALE_EFFECT,
    APPLY_SPELL,
    APPLY_STATUS,
    HEAL,
    REGENERATE,
    CHANGE_COUNT,
    MODIFY_MORALE
};

enum class TargetPattern : uint8_t {
    SELF,
    PRIMARY_TARGET,
    SAME_TARGET,
    LINE_THROUGH_TARGET,
    ALL_ADJACENT_TO_ATTACKER,
    AREA_AROUND_TARGET,
    ALL_ENEMIES
};

struct AbilityObservation {
    uint16_t rawType;
    AbilityTrigger trigger;
    AbilityOperation operation;
    TargetPattern targetPattern;

    uint16_t payloadKind;
    uint16_t payloadId;

    float probability;
    float multiplier;
    float additiveValue;
    float count;
    float radius;
    float duration;

    uint32_t conditionMask;
    uint32_t tagMask;
};
```

This adapter is not learned and does not parse text. It is a small table or `switch` statement tied to the engine’s rule definitions.

Examples:

```text
DOUBLE_SHOOTING
trigger       = ON_RANGED_ATTACK
operation     = REPEAT_ATTACK
targetPattern = SAME_TARGET
count         = 2
probability   = 1
```

```text
MAGIC_RESISTANCE, 25, 0
trigger       = ON_INCOMING_SPELL
operation     = CANCEL_EFFECT
probability   = 0.25
```

```text
SPELL_CASTER, 20, Spell::BLIND
trigger       = ON_HIT
operation     = APPLY_SPELL
targetPattern = PRIMARY_TARGET
payloadKind   = SPELL
payloadId     = BLIND
probability   = 0.20
```

```text
NO_ENEMY_RETALIATION
trigger       = ON_MELEE_ATTACK
operation     = MODIFY_RETALIATION
multiplier    = 0
```

The categorical fields are represented by embeddings. The numerical fields are normalized and passed through an MLP. A token encoder can be written as:

\[
h_k =
\operatorname{MLP}
\left(
E_t[t_k]
\mathbin\Vert
E_g[g_k]
\mathbin\Vert
E_o[o_k]
\mathbin\Vert
E_p[p_k]
\mathbin\Vert
E_v[v_k]
\mathbin\Vert
n_k
\right),
\]

where \(\Vert\) denotes concatenation.

The per-unit ability representation is then:

\[
h_i^{\mathrm{ability}}
=
\operatorname{Pool}
\left(
\{h_{ik}\}_{k=1}^{K_i}
\right).
\]

This representation can generalize to a new creature assembled from familiar operations. A new unit with flying, double attack, and no retaliation can be understood compositionally even if that exact creature ID was withheld during training.

### 4.3 Semantic categories for current fheroes2 abilities

The following grouping is useful when constructing the deterministic adapter.

| Category | Raw ability examples | Recommended semantic fields |
|---|---|---|
| Geometry and movement | `DOUBLE_HEX_SIZE`, `FLYING` | footprint size, movement mode, occupancy geometry |
| Creature tags | `DRAGON`, `UNDEAD`, `ELEMENTAL`, elemental creature tags | tag mask used by conditional damage and spell rules |
| Attack multiplicity | `DOUBLE_SHOOTING`, `DOUBLE_MELEE_ATTACK` | trigger, repeat count, same-target pattern |
| Attack geometry | `TWO_CELL_MELEE_ATTACK`, `ALL_ADJACENT_CELL_MELEE_ATTACK`, `AREA_SHOT` | line, adjacency, radius, affected-cell pattern |
| Retaliation rules | `UNLIMITED_RETALIATION`, `NO_ENEMY_RETALIATION` | retaliation budget or retaliation multiplier |
| Mode-dependent combat | `NO_MELEE_PENALTY` | condition on melee use by a shooter |
| Conditional damage | `DOUBLE_DAMAGE_TO_UNDEAD` and elemental weaknesses | target condition and damage multiplier |
| Spell resistance | general, elemental, fire, cold, mind, and spell-specific immunities or reductions | effect family, payload spell ID, cancel probability, damage multiplier |
| On-hit effects | `SPELL_CASTER` | proc probability and structured spell payload |
| Recovery and growth | `HP_REGENERATION`, `HP_DRAIN`, `SOUL_EATER` | trigger, healed or added amount, cap behavior |
| Global or stochastic effects | `MORAL_DECREMENT`, `ENEMY_HALVING` | scope, probability, magnitude |
| Weakness records | spell or elemental vulnerabilities | incoming effect condition and multiplier |

The raw ability enum should remain available to the model because it can capture residual distinctions. The semantic fields provide the compositional bias. A practical unit encoder uses both:

\[
h_i^{\mathrm{static}}
=
f_{\mathrm{unit}}
\left(
x_i^{\mathrm{stats}},
h_i^{\mathrm{raw\ abilities}},
h_i^{\mathrm{semantic\ abilities}},
E_{\mathrm{creature\ id}}[m_i]
\right).
\]

During training, randomly masking the creature-ID embedding prevents the model from depending exclusively on identity.

---

## 5. Static ability encoding is not sufficient

Many abilities are meaningful only in relation to:

- the current action mode;
- the selected target;
- nearby friendly and enemy units;
- the attacker and defender’s tags;
- the attack direction;
- whether retaliation is available;
- the current HP and count thresholds;
- stochastic proc outcomes.

For example:

- `AREA_SHOT` depends on which units occupy cells around the selected target cell.
- `ALL_ADJACENT_CELL_MELEE_ATTACK` depends on the attacker’s final position.
- `DOUBLE_DAMAGE_TO_UNDEAD` depends on the target’s tag.
- `NO_MELEE_PENALTY` matters only when a shooter is forced into melee.
- `NO_ENEMY_RETALIATION` changes the consequence of a melee action, not merely the attacker’s identity.
- `HP_DRAIN` depends on actual damage and available recovery capacity.
- `ENEMY_HALVING` is stochastic and its value depends on the target stack’s current count.

Therefore the policy should receive both:

\[
\boxed{
\text{static ability semantics}
+
\text{state-conditioned action effects}
}
\]

### 5.1 Engine-grounded candidate effect summary

For every legal action candidate \(a\), compute an effect vector

\[
\psi(s,a)
\]

using the engine’s exact rules or a controlled one-step simulation.

A useful effect vector includes:

\[
\begin{aligned}
\psi(s,a) = [&
\text{movement distance},
\text{destination exposure},
\text{primary target identity},\\
&
\text{affected friendly count},
\text{affected enemy count},\\
&
\mathbb E[\text{enemy HP removed}],
\min[\text{enemy HP removed}],
\max[\text{enemy HP removed}],\\
&
\mathbb E[\text{friendly HP removed}],
\mathbb E[\text{retaliatory HP loss}],\\
&
P(\text{primary target killed}),
P(\text{actor killed}),\\
&
\mathbb E[\text{self-healing}],
\mathbb E[\text{unit-count growth}],\\
&
P(\text{status applied}),
\text{status type},
\text{expected status duration},\\
&
\text{shots consumed},
\text{retaliation consumed},
\text{post-action threat}
].
\end{aligned}
\]

For multi-target attacks, either aggregate by friendly and enemy totals or retain a small set of per-target effect tokens:

\[
\{
(\text{target embedding},\Delta HP,\Delta count,P(\text{status}))
\}_{j\in \mathrm{affected}(s,a)}.
\]

The latter is more expressive and still inexpensive because fheroes2 battles contain few stacks.

This design lets the policy learn:

> “This action is likely to remove a high-value enemy stack with no retaliation and small exposure afterward.”

It does not have to infer the full operational meaning of `NO_ENEMY_RETALIATION` from an opaque label.

### 5.2 Three implementation levels

#### Level A: analytic summaries

Use engine helper functions for potential damage, retaliation estimates, reachability, attack geometry, and affected cells. The current battle unit interface already exposes minimum and maximum damage, potential damage, and an approximate retaliation estimate.[^battle-unit]

This is the fastest option and is suitable for the first baseline.

#### Level B: one-step dry runs

Clone or serialize the current battle state, execute one candidate, and compare the resulting state:

\[
\psi(s,a)
=
F(T(s,a))-F(s).
\]

For stochastic effects, execute several one-step samples or enumerate small discrete branches.

This is more reliable for complicated abilities because it reuses the same transition code as the game.

#### Level C: limited lookahead

Continue the dry run for one or more future unit turns using the current policy or a value model:

\[
\widehat Q(s,a)
=
\frac{1}{N}
\sum_{k=1}^{N}
\left[
R_{0:H}^{(k)}
+
V_\theta(s_H^{(k)})
\right].
\]

This produces the strongest decision rule, at greater computational cost.

### 5.3 Low-latency inference

None of the above requires an LLM. The battle is turn-based, so the model is called only at decision points.

A low-latency implementation should:

1. cache every creature’s static stat and ability embedding;
2. update only dynamic stack features each turn;
3. enumerate legal actions in C++;
4. compute candidate effect summaries in C++;
5. batch all legal candidates into one network forward pass;
6. use a small entity encoder and candidate scorer;
7. reserve expensive dry-run search for close or uncertain decisions.

The number of stack entities is small, and the battlefield has only 99 cells. A compact model with a few attention layers is sufficient. The ability representation adds negligible latency compared with rendering or simulation.

---

## 6. Unit-centric state representation

Each living stack should be represented as one entity token. The same unit encoder is applied to every stack.

### 6.1 Static creature features

A stack token should contain rule-relevant creature properties:

\[
x_i^{\mathrm{static}}
=
[
\text{attack},
\text{defense},
d_{\min},
d_{\max},
\text{HP per creature},
\text{speed},
\text{base shots},
\text{base strength},
\text{level},
\text{cost features}
].
\]

Append the pooled ability and weakness representations from Section 4.

The repository’s `MonsterBattleStats` already contains attack, defense, minimum and maximum damage, HP, speed, shots, base strength, abilities, and weaknesses.[^monster-info-header]

### 6.2 Dynamic stack features

For the current battle state, include:

\[
x_i^{\mathrm{dynamic}}
=
[
\text{side},
\text{current count},
\text{total HP},
\text{top creature HP},
\text{shots remaining},
\text{head cell},
\text{tail cell},
\text{orientation},
\text{has acted},
\text{retaliation state},
\text{turn-order position},
\text{status effects}
].
\]

Status effects should also be structured records, for example:

```json
{
  "effect_type_id": 12,
  "remaining_rounds": 2,
  "source_side": 0,
  "magnitude": 0.25
}
```

Do not concatenate status names into a text string.

### 6.3 Absolute and normalized numerical features

Use both absolute and relative values. For stack count, a useful group is:

\[
[
\log(1+n_i),
n_i,
n_i / \sum_{j\in\mathrm{friendly}}n_j,
\text{stack value}_i / \text{friendly army value}
].
\]

Absolute quantities preserve discrete kill thresholds. Relative quantities improve transfer across army scales.

For HP, include:

\[
[
\text{total HP},
\log(1+\text{total HP}),
\text{HP fraction},
\text{top-unit HP fraction}
].
\]

Do not normalize away all absolute scale. Whether an attack kills the last creature before retaliation is an important discontinuity.

### 6.4 Pairwise relations

For each relevant pair of stacks \(i,j\), compute:

\[
r_{ij}
=
[
\text{hex distance},
\text{same side},
\text{can }i\text{ reach }j,
\text{can }j\text{ reach }i,
\text{expected damage }i\to j,
\text{expected retaliation }j\to i,
\text{kill probability},
\text{relative initiative},
\text{line or splash relationship}
].
\]

These can be used as attention biases or edge features.

The reusable tactical concept is not “attack a specific creature ID.” It is closer to:

> “This fragile, high-damage stack can be eliminated before it acts, with limited retaliation and acceptable follow-up exposure.”

A set-based entity encoder is appropriate because stack serialization order should not change the decision. Set Transformer provides one established attention-based architecture for permutation-invariant set inputs.[^set-transformer]

---

## 7. Board representation

The board can be represented in either of two ways.

### 7.1 Explicit cell tokens

Create one token for each of the 99 cells:

\[
c_k
=
[
x_k,
y_k,
\text{passability},
\text{obstacle type},
\text{occupant},
\text{threat},
\text{reachability},
\text{deployment side}
].
\]

Use fixed hex-grid adjacency as graph edges, or use full attention with relative-coordinate biases.

This is the most general approach.

### 7.2 Candidate-attached board features

For a smaller initial model, avoid a full board encoder. Attach relevant spatial features to each legal candidate:

- destination cell;
- distance traveled;
- cells occupied by a wide unit;
- adjacency to each stack;
- number of enemies able to reach the destination;
- whether the destination blocks a shooter;
- whether the destination is inside an area threat;
- attack direction and affected-cell pattern.

This often suffices because the engine already performs pathfinding and legal-action generation.

A reasonable progression is to begin with candidate-attached spatial features and add explicit cell tokens only if the policy fails on obstacle-sensitive positioning.

---

## 8. Legal candidate-action scoring

The model should not generate arbitrary low-level commands. The engine should enumerate complete legal macro-actions.

Examples are:

```text
SKIP
MOVE(destination_cell)
RANGED_ATTACK(target_stack, target_cell)
MELEE_ATTACK(target_stack, attack_from_cell, attack_direction)
```

If hero spells are later enabled:

```text
CAST_SPELL(spell_id, target_cell_or_unit)
```

A melee candidate should specify a complete valid geometry. The policy should not independently generate target, movement cell, and direction and then hope that the combination is legal.

### 8.1 Candidate schema

```json
{
  "candidate_id": 37,
  "type_id": 2,
  "actor_index": 1,
  "target_index": 6,
  "destination_cell": 43,
  "attack_cell": 44,
  "attack_direction": 4,
  "affected_unit_indices": [6, 8],
  "effect_summary": {
    "expected_enemy_hp_removed": 57.3,
    "expected_friendly_hp_removed": 0.0,
    "expected_retaliation_hp_loss": 0.0,
    "primary_kill_probability": 0.82,
    "expected_self_heal": 12.4
  }
}
```

The policy returns only:

```json
{
  "candidate_id": 37
}
```

The engine converts that candidate to its internal `Battle::Command` sequence.

### 8.2 Candidate-scoring model

Let:

- \(h_i\) be the contextualized actor embedding;
- \(h_j\) be the target embedding;
- \(g_k\) be the destination-cell embedding;
- \(h_{\mathrm{global}}\) be a pooled battle embedding;
- \(\psi(s,a)\) be the engine-grounded effect summary.

Construct:

\[
z_a =
f_{\mathrm{candidate}}
\left(
E_{\mathrm{type}}[\tau_a],
h_i,
h_j,
g_k,
r_{ij},
\psi(s,a),
h_{\mathrm{global}}
\right).
\]

Score each legal candidate:

\[
\ell_a=f_{\mathrm{score}}(z_a),
\]

then normalize only over the legal candidate set:

\[
\pi_\theta(a\mid s)
=
\frac{\exp(\ell_a)}
{\sum_{a'\in\mathcal A(s)}\exp(\ell_{a'})}.
\]

This architecture naturally supports a variable number of actions and different army compositions.

---

## 9. Objective: win first, then minimize permanent losses

A scalar reward such as

\[
R
=
\mathbf 1\{\mathrm{win}\}
-
\lambda \cdot \mathrm{losses}
\]

can behave poorly. If \(\lambda\) is too large, the agent may choose a low-casualty defeat over a necessary but costly action that preserves a chance to win.

Use a lexicographic objective:

\[
J_1(\pi)=\Pr_\pi(\mathrm{victory}),
\]

\[
J_2(\pi)
=
-\mathbb E_\pi[L\mid \mathrm{victory}].
\]

The policy first maximizes victory probability. Among actions with essentially equal victory probability, it minimizes losses.

### 9.1 Permanent-loss measure

Let \(v_i\) be the strategic value of one creature in the initial friendly stack \(i\). Define:

\[
L
=
\frac{
\sum_i v_i
\max(0,n_{i,0}-n_{i,T}^{\mathrm{retained}})
}{
\sum_i v_i n_{i,0}
}.
\]

The `max` is important for growth abilities. New creatures created by `SOUL_EATER` or similar mechanics should not automatically cancel losses in unrelated original stacks unless the strategic objective explicitly values net army growth.

If post-battle strategic value matters, track a separate metric:

\[
G
=
\frac{V_{\mathrm{final\ army}}-V_{\mathrm{initial\ army}}}
{V_{\mathrm{initial\ army}}}.
\]

Do not silently combine permanent losses and generated-unit gains into one ambiguous reward.

### 9.2 Two value heads

Train:

\[
V_{\mathrm{win}}(s)
=
P(\mathrm{victory}\mid s),
\]

and:

\[
V_{\mathrm{loss}}(s)
=
\mathbb E[L\mid s,\mathrm{victory}].
\]

For candidate actions:

\[
Q_{\mathrm{win}}(s,a),
\qquad
Q_{\mathrm{loss}}(s,a).
\]

At inference:

\[
p_{\max}=\max_a Q_{\mathrm{win}}(s,a),
\]

\[
\mathcal A_\epsilon
=
\{a:Q_{\mathrm{win}}(s,a)\geq p_{\max}-\epsilon\},
\]

\[
a^*
=
\arg\min_{a\in\mathcal A_\epsilon}
Q_{\mathrm{loss}}(s,a).
\]

The tolerance \(\epsilon\) should be calibrated from held-out games and confidence intervals rather than chosen solely by intuition.

---

## 10. Scenario generation

Uniformly sampling creature IDs and counts creates many trivial battles. The agent can learn to recognize overwhelming strength rather than tactics.

Sample an approximate army-value budget first.

For one side:

\[
K\sim\operatorname{Uniform}\{1,\ldots,K_{\max}\},
\]

\[
B\sim\operatorname{LogUniform}(B_{\min},B_{\max}),
\]

\[
w\sim\operatorname{Dirichlet}(\alpha\mathbf 1),
\]

where:

- \(K\) is the number of stacks;
- \(B\) is total army budget;
- \(w_i\) allocates that budget among stacks.

For selected creature types \(m_i\), choose:

\[
n_i
=
\max
\left(
1,
\operatorname{round}
\frac{B w_i}{v(m_i)}
\right).
\]

For the opponent:

\[
B_{\mathrm{enemy}}=rB_{\mathrm{self}}.
\]

Sample \(r\) from a mixture concentrated near one:

\[
\log r
\sim
\eta\mathcal N(0,\sigma_{\mathrm{close}}^2)
+
(1-\eta)\mathcal N(0,\sigma_{\mathrm{wide}}^2).
\]

This produces many close battles and some favorable or unfavorable battles.

### 10.1 Stratify by mechanics

Track coverage of ability families:

- movement and wide-unit geometry;
- ranged attacks;
- repeat attacks;
- retaliation modifications;
- multi-target attacks;
- spell resistance;
- on-hit status effects;
- healing and regeneration;
- stochastic count-changing abilities.

Without stratification, common ordinary melee units can dominate the dataset.

### 10.2 Curriculum

A suitable curriculum is:

1. randomized one-stack battles with many creature types;
2. two-stack battles emphasizing focus fire and turn order;
3. three-stack battles emphasizing protection, blocking, and sacrifice;
4. one-to-five-stack battles;
5. special mechanics and terrain;
6. hero spells and sieges, if desired.

Earlier stages should remain in the replay mixture. The curriculum changes sampling weights; it should not permanently discard simple battles.

### 10.3 Adaptive scenario sampling

Increase the probability of scenarios where:

- estimated win probability is neither near zero nor one;
- the policy and search disagree;
- recent policy checkpoints disagree;
- the value model is poorly calibrated;
- action entropy is high;
- held-out mechanic performance is weak.

These battles provide more learning signal than repeated trivial victories.

---

## 11. Training sequence

Starting from pure reinforcement learning is possible but inefficient. The current engine already contains a battle planner that produces coherent actions and evaluates concepts such as threat, attack positions, area attacks, and retaliation. It is useful as a bootstrap teacher, not as an optimal oracle.[^ai-battle]

### 11.1 Phase A: behavior cloning

Generate randomized battles with the built-in AI and record:

\[
(s_t,\mathcal A(s_t),a_t^{\mathrm{AI}},z),
\]

where \(z\) contains final outcome and losses.

Train:

\[
\mathcal L_{\mathrm{BC}}
=
-\log\pi_\theta(a_t^{\mathrm{AI}}\mid s_t).
\]

Add value and auxiliary losses:

\[
\mathcal L
=
\mathcal L_{\mathrm{BC}}
+
\lambda_w\mathcal L_{\mathrm{win}}
+
\lambda_l\mathcal L_{\mathrm{loss}}
+
\lambda_d\mathcal L_{\mathrm{damage}}
+
\lambda_r\mathcal L_{\mathrm{retaliation}}
+
\lambda_e\mathcal L_{\mathrm{effect}}.
\]

Useful auxiliary targets include:

- immediate damage distribution;
- retaliation damage;
- affected-unit set;
- kill probability;
- status-proc probability;
- reachability;
- next actor;
- final outcome.

These teach the representation without modifying the true objective.

### 11.2 Phase B: DAgger-style data aggregation

Pure behavior cloning sees the teacher’s state distribution. Once the learned policy makes a mistake, it can encounter unfamiliar states.

Run the learned policy, query the built-in AI on the states the learner reaches, and aggregate:

\[
D_{k+1}
=
D_k
\cup
\{(s,a_{\mathrm{AI}}):s\sim d_{\pi_k}\}.
\]

Prioritize disagreement states. DAgger was designed to address this distribution-shift problem in sequential imitation learning.[^dagger]

The built-in AI remains an imperfect labeler. Its role here is to restore basic competence in off-distribution states, not define the final optimum.

### 11.3 Phase C: rollout or search distillation

For selected states:

1. snapshot the battle state and random-generator state;
2. enumerate legal candidates;
3. apply each candidate in separate branches;
4. complete the battle with the current policy, built-in AI, or a mixture;
5. estimate victory probability and conditional losses;
6. produce a soft action target.

For example:

\[
p_{\mathrm{search}}(a\mid s)
\propto
\exp
\left(
\frac{
\widehat Q_{\mathrm{lex}}(s,a)
}{\tau}
\right).
\]

Distill it with:

\[
\mathcal L_{\mathrm{search}}
=
-\sum_a
p_{\mathrm{search}}(a\mid s)
\log\pi_\theta(a\mid s).
\]

Preserve the engine RNG state when branching. In the current battle loop, command contents are used to update the PCG stream, so action choice and random-stream evolution are connected.[^arena][^command]

### 11.4 Phase D: self-play with an opponent population

Training only against the built-in AI can produce a narrow best response to its habits.

Use a mixture of opponents:

- current policy;
- historical checkpoints;
- built-in AI;
- scripted tactical policies;
- search-enhanced policy.

A shared network can control either side by canonicalizing the observation so that the acting side is always “self.”

A population reduces cyclic forgetting and encourages robust tactics. Large game agents such as AlphaStar and OpenAI Five also used diverse or historical opponent populations, although their scale and exact algorithms need not be copied.[^alphastar][^openai-five]

### 11.5 Phase E: optional RL fine-tuning

After imitation and search distillation produce a competent policy, use PPO or another actor-critic method for additional improvement.

RL should not be expected to repair a poor observation schema. It is most useful after:

- legal actions are correct;
- ability effects are represented;
- scenario coverage is broad;
- reward priorities are unambiguous;
- the policy has basic competence.

---

## 12. Search for stronger play

A network-only policy is an approximation. Since the exact simulator is available, the strongest practical agent is:

\[
\boxed{
\text{structured policy and value model}
+
\text{exact simulator}
+
\text{limited search}
}
\]

For small battles, use exhaustive or high-budget expectimax to produce approximate oracle labels.

For larger battles, use:

- top-\(K\) policy candidates;
- sampled chance outcomes;
- one or two rounds of lookahead;
- value-network evaluation at the frontier.

A simple form is:

\[
Q(s,a)
\approx
\frac{1}{N}
\sum_{k=1}^{N}
\left[
R_{0:H}^{(k)}
+
V_\theta(s_H^{(k)})
\right].
\]

Even short lookahead can capture many important fheroes2 decisions because retaliation, kill thresholds, area effects, and next-turn reachability often determine the immediate tactical tradeoff.

The network should not observe hidden RNG state. Search can preserve and branch simulator state internally, but exposing the hidden random stream to the policy would create clairvoyant behavior.

---

## 13. Generalization evaluation

Randomly splitting individual battle states is not sufficient. States from closely related trajectories can appear in both training and test data.

Assign entire initial-scenario families to splits before generating trajectories.

| Split | Withheld information | Purpose |
|---|---|---|
| Seed holdout | battle and obstacle seeds | ordinary robustness |
| Pairing holdout | selected creature-versus-creature edges | new matchup combinations |
| Composition holdout | entire army multisets | combinatorial generalization |
| Count holdout | count or value ranges | numerical extrapolation |
| Creature holdout | selected creature IDs | stat and ability transfer |
| Ability-combination holdout | combinations of familiar primitives | compositional ability reasoning |
| Mechanic holdout | an entire operation family | dependence on explicit semantics or simulation |
| Opponent holdout | policies absent from training | policy robustness |

### 13.1 Clean one-stack pairing split

Represent creature types as graph vertices. A one-stack matchup is an edge.

Construct train and test edges so that:

- every creature appears in training;
- selected pairings are absent from training;
- all test trajectories come from held-out edges.

This tests whether the policy learned relational combat reasoning rather than a complete matchup table.

### 13.2 Ability-focused tests

Create controlled counterfactual pairs with nearly identical statistics but different abilities.

Examples:

- ordinary ranged attack versus double shooting;
- retaliation versus no retaliation;
- ordinary shooter versus area shot;
- ordinary melee unit versus life drain;
- one-hex versus two-hex geometry.

Measure whether the chosen action changes in the expected direction.

### 13.3 Ablation suite

Train and compare:

1. creature ID only;
2. numerical stats only;
3. stats plus raw ability type embeddings;
4. stats plus semantic ability tokens;
5. semantic tokens plus engine-grounded candidate effects;
6. candidate effects plus search.

This isolates which component produces generalization.

### 13.4 Metrics

Report at least:

\[
\left(
P(\mathrm{win}),
\mathbb E[L\mid\mathrm{win}]
\right).
\]

Additional metrics:

- action regret against a small-battle search oracle;
- win-probability calibration;
- loss-value calibration;
- cross-play matrix against opponent checkpoints;
- side-reflection consistency;
- composition-holdout performance;
- latency per decision;
- invalid-action rate, which should be zero with legal candidate scoring.

For stochastic battles, evaluate each scenario across multiple random seeds and report confidence intervals.

---

## 14. Suggested observation and action protocol

The following JSON is intended as a transport and debugging format. The online model should receive tensors generated from it, not serialized text tokens.

```json
{
  "schema_version": 1,
  "battle_id": 8124,
  "round": 3,
  "acting_side": 0,
  "active_unit_index": 1,

  "units": [
    {
      "unit_index": 1,
      "side": 0,
      "creature_id": 24,

      "static": {
        "attack": 5,
        "defense": 3,
        "damage_min": 2,
        "damage_max": 3,
        "hp_per_creature": 15,
        "speed": 5,
        "base_shots": 0
      },

      "dynamic": {
        "count": 18,
        "total_hp": 263,
        "top_creature_hp": 8,
        "shots_left": 0,
        "head_cell": 31,
        "tail_cell": -1,
        "reflected": false,
        "has_acted": false,
        "retaliation_available": true
      },

      "abilities": [
        {
          "raw_type_id": 2,
          "raw_percentage": 0,
          "raw_value": 0,

          "trigger_id": 0,
          "operation_id": 2,
          "target_pattern_id": 0,
          "payload_kind_id": 0,
          "payload_id": 0,

          "probability": 1.0,
          "multiplier": 1.0,
          "additive_value": 0.0,
          "count": 0.0,
          "radius": 0.0,
          "duration": 0.0
        }
      ],

      "weaknesses": [],
      "status_effects": []
    }
  ],

  "cells": [
    {
      "cell_index": 0,
      "x": 0,
      "y": 0,
      "passable": true,
      "obstacle_type_id": 0,
      "occupant_unit_index": -1
    }
  ],

  "turn_order": [1, 7, 3, 9],

  "legal_candidates": [
    {
      "candidate_id": 37,
      "type_id": 2,
      "actor_index": 1,
      "target_index": 7,
      "destination_cell": 43,
      "attack_cell": 44,
      "attack_direction": 4,
      "affected_unit_indices": [7],

      "effect_summary": {
        "enemy_hp_removed_min": 40,
        "enemy_hp_removed_mean": 53.5,
        "enemy_hp_removed_max": 67,
        "friendly_hp_removed_mean": 0.0,
        "retaliation_hp_loss_mean": 18.2,
        "primary_kill_probability": 0.35,
        "actor_kill_probability": 0.0,
        "self_heal_mean": 0.0,
        "status_type_id": 0,
        "status_probability": 0.0
      }
    }
  ]
}
```

### 14.1 C++ adapter sketch

```cpp
AbilityObservation encodeAbility( const fheroes2::MonsterAbility & ability )
{
    AbilityObservation out{};
    out.rawType = static_cast<uint16_t>( ability.type );
    out.probability = 1.0f;

    switch ( ability.type ) {
    case fheroes2::MonsterAbilityType::DOUBLE_SHOOTING:
        out.trigger = AbilityTrigger::ON_RANGED_ATTACK;
        out.operation = AbilityOperation::REPEAT_ATTACK;
        out.targetPattern = TargetPattern::SAME_TARGET;
        out.count = 2.0f;
        break;

    case fheroes2::MonsterAbilityType::MAGIC_RESISTANCE:
        out.trigger = AbilityTrigger::ON_INCOMING_SPELL;
        out.operation = AbilityOperation::CANCEL_EFFECT;
        out.targetPattern = TargetPattern::SELF;
        out.probability = static_cast<float>( ability.percentage ) / 100.0f;
        break;

    case fheroes2::MonsterAbilityType::SPELL_CASTER:
        out.trigger = AbilityTrigger::ON_HIT;
        out.operation = AbilityOperation::APPLY_SPELL;
        out.targetPattern = TargetPattern::PRIMARY_TARGET;
        out.payloadKind = static_cast<uint16_t>( PayloadKind::SPELL );
        out.payloadId = static_cast<uint16_t>( ability.value );
        out.probability = static_cast<float>( ability.percentage ) / 100.0f;
        break;

    case fheroes2::MonsterAbilityType::NO_ENEMY_RETALIATION:
        out.trigger = AbilityTrigger::ON_MELEE_ATTACK;
        out.operation = AbilityOperation::MODIFY_RETALIATION;
        out.targetPattern = TargetPattern::PRIMARY_TARGET;
        out.multiplier = 0.0f;
        break;

    default:
        // Every enum value should be handled explicitly.
        // Unknown values should fail closed in training builds.
        break;
    }

    return out;
}
```

The string names remain useful in logs, but the model consumes the numeric tensor fields.

---

## 15. Recommended initial implementation

### 15.1 Initial scope

Begin with:

- field battles;
- creature stacks only;
- no hero spells;
- no siege structures;
- no retreat or surrender;
- one to three stacks per side;
- a stratified subset of creature mechanics;
- headless simulation.

This retains the core tactical problem while keeping the state and action schemas manageable.

### 15.2 Model

A practical small model can use:

- static unit MLP;
- ability-token MLP and sum or attention pooling;
- dynamic unit MLP;
- three or four entity self-attention layers;
- optional lightweight cell encoder;
- relation features as attention biases;
- candidate-action MLP;
- policy head;
- win-probability head;
- conditional-loss head.

With at most a small number of stack tokens, this model is inexpensive.

### 15.3 Training progression

Use:

\[
\text{behavior cloning}
\rightarrow
\text{DAgger}
\rightarrow
\text{one-step rollout distillation}
\rightarrow
\text{self-play population}
\rightarrow
\text{optional RL and deeper search}.
\]

Do not begin with five Peasants against five Peasants as the only training distribution. Retain it as an integration test. The first learning distribution should already randomize creature types, counts, sides, and seeds.

### 15.4 First meaningful experiment

A useful first experiment is:

1. choose a creature subset covering ordinary melee, ranged, flying, double attack, no retaliation, and one area attack;
2. train on randomized one-stack and two-stack battles;
3. hold out selected creature pairings;
4. compare raw ability embeddings against semantic tokens;
5. add engine-grounded candidate effects;
6. measure held-out win rate and action regret;
7. verify side-reflection consistency;
8. verify zero invalid actions.

This directly tests whether the ability representation contributes to composition generalization.

---

## 16. Common failure modes

### 16.1 Creature-ID memorization

**Symptom:** strong random-split performance and poor held-out-pairing performance.

**Mitigation:** semantic stats and abilities, ID masking, composition-level splits.

### 16.2 Treating enum IDs as ordered numbers

**Symptom:** unstable or meaningless interpolation between unrelated ability types.

**Mitigation:** one-hot or embedding lookup for categorical fields.

### 16.3 One static ability vector without action context

**Symptom:** the policy knows a unit has `AREA_SHOT` but repeatedly chooses target cells that damage friendly stacks.

**Mitigation:** engine-computed affected-unit and outcome features for every candidate.

### 16.4 Losing repeated or parameterized abilities

**Symptom:** spell-specific immunities or multiple payload records collapse into one bit.

**Mitigation:** encode abilities as a variable-length set of records, not one aggregate flag per type.

### 16.5 Rewarding low-casualty defeats

**Symptom:** the policy avoids necessary engagements and preserves units while losing.

**Mitigation:** lexicographic victory-first objective and separate value heads.

### 16.6 Trivial scenario distribution

**Symptom:** high average win rate but poor performance on close battles.

**Mitigation:** army-budget matching, strength-ratio mixture, adaptive hard-scenario sampling.

### 16.7 Teacher overfitting

**Symptom:** the learned agent reproduces built-in AI weaknesses.

**Mitigation:** search distillation, self-play population, oracle labels for small battles.

### 16.8 Hidden RNG leakage

**Symptom:** implausibly precise decisions that exploit future random outcomes.

**Mitigation:** preserve RNG state for simulation, but do not expose it to the policy observation.

### 16.9 Claiming zero-shot support for an unseen primitive

**Symptom:** a new ability receives a new ID and the policy is expected to understand it.

**Mitigation:** describe it using known semantic operations or let the simulator produce candidate consequences. A genuinely unseen operation cannot be inferred from an arbitrary identifier.

---

## 17. Implementation milestones

### Milestone 1: deterministic environment adapter

Deliverables:

- headless battle launch;
- deterministic scenario seed;
- complete observation dump;
- legal candidate enumeration;
- candidate-to-command conversion;
- battle result and permanent-loss accounting;
- replayable trajectories.

Tests:

- repeated seed produces identical trajectory;
- every candidate executes without engine rejection;
- side reflection preserves equivalent outcomes;
- RNG state is restored correctly after branching.

### Milestone 2: structured state and abilities

Deliverables:

- unit tokens;
- raw ability and weakness records;
- semantic ability adapter;
- status-effect records;
- normalized numerical features;
- tensor conversion.

Tests:

- every current ability enum has an explicit adapter case;
- raw and semantic records reconstruct expected rule metadata;
- localized UI language does not change observations;
- repeated ability types remain distinct.

### Milestone 3: legal-action behavioral-cloning baseline

Deliverables:

- demonstration generation from built-in AI;
- entity encoder;
- candidate scorer;
- policy and value losses;
- random-seed and pairing holdouts.

Acceptance criteria:

- zero invalid actions;
- competent in-distribution play;
- nontrivial held-out-pairing performance;
- comparable behavior under side reflection.

### Milestone 4: candidate effect summaries

Deliverables:

- immediate damage and retaliation features;
- affected-unit sets;
- status-proc and recovery features;
- one-step dry-run option;
- effect-prediction auxiliary heads.

Acceptance criteria:

- improved area-attack friendly-fire decisions;
- improved no-retaliation and life-drain decisions;
- lower action regret on small battles;
- bounded inference latency.

### Milestone 5: search distillation and self-play

Deliverables:

- battle-state snapshot and clone;
- chance sampling;
- top-\(K\) candidate search;
- search policy targets;
- historical opponent pool;
- cross-play evaluation.

Acceptance criteria:

- exceeds built-in AI on held-out scenario families;
- does not regress substantially against historical checkpoints;
- improves conditional loss at matched win probability.

---

## 18. Central recommendation

For the current fheroes2 codebase, ability text should not be part of the learning interface. The model should use:

\[
\boxed{
\text{raw typed ability records}
}
\]

plus:

\[
\boxed{
\text{semantic primitive tokens}
}
\]

plus:

\[
\boxed{
\text{engine-grounded, candidate-specific consequences}
}
\]

The first component is faithful to the repository. The second supports compositional transfer across creatures. The third handles context-dependent interactions and sharply reduces the amount of rule inference the neural network must learn.

The full generalized agent is therefore:

\[
\boxed{
\text{entity-centric battle state}
+
\text{structured ability set}
+
\text{legal candidate-action scoring}
+
\text{scenario diversity}
+
\text{simulator-guided policy improvement}
}
\]

This design remains a conventional low-latency neural battle agent. It does not convert combat into text, and it does not require an LLM at inference time.

---

## References

[^monster-info-header]: fheroes2, [`monster_info.h`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/monster/monster_info.h). Defines `MonsterAbilityType`, `MonsterWeaknessType`, `MonsterAbility`, `MonsterWeakness`, and `MonsterBattleStats`.

[^monster-info-source]: fheroes2, [`monster_info.cpp`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/monster/monster_info.cpp). Populates creature abilities and separately maps structured ability records to localized descriptions.

[^monster-assignments]: fheroes2, [`monster_info.cpp`, ability assignments](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/monster/monster_info.cpp#L371-L486). Includes parameterless abilities, percentage-valued resistance, and spell-caster records carrying spell IDs.

[^board]: fheroes2, [`battle_board.h`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/battle/battle_board.h). Defines the 11 by 9 battlefield and board geometry helpers.

[^arena]: fheroes2, [`battle_arena.cpp`](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_arena.cpp#L423-L500). Shows the unit-turn loop, built-in AI planner call, command application, and PCG stream update.

[^command]: fheroes2, [`battle_command.h`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/battle/battle_command.h) and [`battle_command.cpp`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/battle/battle_command.cpp). Define command types, attack parameters, and command-dependent random-stream updates.

[^battle-unit]: fheroes2, [`battle_troop.h`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/battle/battle_troop.h). Exposes dynamic unit state, minimum and maximum damage, potential damage, retaliation estimates, spell interactions, and battle status.

[^ai-battle]: fheroes2, [`ai_battle.cpp`](https://raw.githubusercontent.com/ihhub/fheroes2/master/src/fheroes2/ai/ai_battle.cpp). Implements the built-in battle planner, including target threat, attack positions, area attacks, and tactical heuristics.

[^set-transformer]: Juho Lee et al., [“Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks”](https://arxiv.org/abs/1810.00825), ICML 2019.

[^dagger]: Stéphane Ross, Geoffrey Gordon, and Drew Bagnell, [“A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning”](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.

[^alphastar]: Oriol Vinyals et al., [“Grandmaster level in StarCraft II using multi-agent reinforcement learning”](https://www.nature.com/articles/s41586-019-1724-z), Nature 2019.

[^openai-five]: OpenAI et al., [“Dota 2 with Large Scale Deep Reinforcement Learning”](https://arxiv.org/abs/1912.06680), 2019.
