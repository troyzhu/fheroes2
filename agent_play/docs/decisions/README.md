---
title: Decision records
type: moc
updated: 2026-07-30
tags: [adr, decisions, agent-env]
---

# Decision records

Each record states a decision that binds implementation, the evidence behind it, and what it amends in the original specification. Where a record and the specification disagree, the record wins, since every record was written later and against verified evidence.

Every record follows the same shape. A sub-problem section says exactly which question is being answered and which are out of scope. An options table lists what was considered, with the argument for and against each. A why-this-one section names the trade-off that decided it and the cost accepted. The header states whether the decision is built, with the file that proves it, because a record and the code can drift and the record should say which side of that gap it is on.

Evidence from other systems follows the same rule as the rest of the tree: the system is named and its note under `research/works/` is linked at first use, so unlabeled prose is always about this project and a claim about another game is checkable at the mention.

| Record | Decision | Amends | Accepted | Built |
|---|---|---|---|---|
| [[0001-observation-profiles]] | Two observability profiles, `full_v1` and `observable_v1`, over one schema, with the digest always covering full state | spec §12 | 2026-07-27 | Partly: the `full_v1` emitter was built early (`agent_observation.{h,cpp}`); `observable_v1` binds Milestone 4 |
| [[0002-action-space]] | A fixed canonical action space with a legality mask, both derived from the same engine enumeration as the candidate list | spec §10.4 | 2026-07-27 | Yes, `agent_action_space.h`, 793 slots, `verify_m3.sh` 8/8 |
| [[0003-config-management]] | Versioned YAML configuration with strict schemas; every artifact embeds its resolved configuration, that configuration's hash, and the commit | spec §11, §15 | 2026-07-27 | No, binds Milestone 4. `configs/` does not exist yet |
| [[0004-spatial-observation-modality]] | An optional semantic `planes_v1` modality; rendered pixels permanently excluded from the training environment | spec §12 | 2026-07-29 | No, plane emitter lands at Milestone 4 |
| [[0005-training-and-reward]] | Imitation first, then masked PPO against an opponent mixture. Reward deliberately open, with candidates and choice criteria fixed | spec §17, §21 | algorithm 2026-07-30, reward open | Partly: cloning, critic pre-fitting and masked PPO exist (`python/fheroes2_agent/`, `verify_agent.sh`); DAgger and every reward candidate but the margin-weighted terminal one remain open |
| [[0006-encoding-count-scaling]] | Counts and hit points log-scaled in the observation encoding; invisible in range, worth 24 standard errors under count extrapolation, which is the real-map regime | amends 0001 in one constant | 2026-08-05 | Yes, `obs_encoding_v3`, clone retrained |

Two records are fully built, two partly, and two not at all, and the column is the per-record ground truth. It exists so that a reader never has to guess whether a record describes the code or describes an intention, which was a real ambiguity before 2026-07-31; [[../implementation/inventory]] carries the component-level detail behind every "built" claim.

Records 0001 and 0004 together define the observation interface, and 0002 defines the action interface. Read those three before implementing the protocol. Records 0003 and 0005 bind from the training milestones onward, and 0005 is the one to read for how a policy will actually be trained and what it will be rewarded for.

## Related

- [[../README|Tree README]], the routing table.
- [[../implementation/README|Implementation]], how the decided mechanisms actually work.
- [[../research/findings|Research findings]], the evidence these records rest on.
