---
title: Decision records
type: moc
updated: 2026-07-30
tags: [adr, decisions, agent-env]
---

# Decision records

Each record states a decision that binds implementation, the evidence behind it, and what it amends in the original specification. Where a record and the specification disagree, the record wins, since every record was written later and against verified evidence.

| Record | Decision | Amends | Status |
|---|---|---|---|
| [[0001-observation-profiles]] | Two observability profiles, `full_v1` and `observable_v1`, over one schema, with the digest always covering full state | spec §12 | accepted 2026-07-27 |
| [[0002-action-space]] | A fixed canonical action space with a legality mask, both derived from the same engine enumeration as the candidate list | spec §10.4 | accepted 2026-07-27 |
| [[0003-config-management]] | Versioned YAML configuration with strict schemas; every artifact embeds its resolved configuration, that configuration's hash, and the commit | spec §11, §15 | accepted 2026-07-27 |
| [[0004-spatial-observation-modality]] | An optional semantic `planes_v1` modality; rendered pixels permanently excluded from the training environment | spec §12 | accepted 2026-07-29 |
| [[0005-training-and-reward]] | Imitation first, then masked PPO against an opponent mixture. Reward deliberately open, with candidates and choice criteria fixed | spec §17, §21 | algorithm accepted, reward open, 2026-07-30 |

Records 0001 and 0004 together define the observation interface, and 0002 defines the action interface. Read those three before implementing the protocol. Records 0003 and 0005 bind from the training milestones onward, and 0005 is the one to read for how a policy will actually be trained and what it will be rewarded for.

## Related

- [[../README|Tree README]], the routing table.
- [[../implementation/README|Implementation]], how the decided mechanisms actually work.
- [[../research/findings|Research findings]], the evidence these records rest on.
