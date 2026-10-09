# Publication changes

The historical counts describe the original run, before these release changes. No new screening or clustering result is claimed.

- Bind checkpoints to the exact input workbook and stage script hashes. Reject missing manifests and changed inputs/configuration.
- Validate every clustering column used downstream and reject missing/duplicate IDs and empty embedding text.
- Reject non-finite or zero embeddings before clustering.
- Reject fractional/boolean relevance scores instead of silently coercing them.
- Validate theme strings and list entries; write theme workbooks atomically.
- Stop on unreadable theme checkpoints instead of overwriting them.
- Describe thematic synthesis as literature mapping in the prompt.

Original scripts remain unchanged in the researcher's local archive. Their original hashes are retained in `provenance.json`.
