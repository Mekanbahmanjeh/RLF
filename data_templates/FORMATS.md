# Audited Training Data Formats

`solstice audit-data` and `solstice train-data` consume an immutable tab-separated
ledger. Blank lines and lines beginning with `#` are ignored. Tabs inside field
values are not supported.

## Ledger

The first non-comment row must contain these exact columns:

```text
shard_id	kind	split	modality	language	domain	format	path	source_uri	license	created_utc	sha256	preprocessing_version	teacher	evaluation_family	approved
```

Supported values:

- `kind`: `text`, `dialogue`, `instruction`, `preference`, `vision`, `video`,
  `tools`, `facts`, or `rules`.
- `split`: `train`, `development`, or `evaluation`.
- `format`: `text_lines`, `tsv`, `vision_tsv`, `video_frames_tsv`, or `binary`.
- `approved`: `true` or `false`.

Paths are resolved relative to the ledger. Every shard must bind its source,
license, creation date, expected SHA-256 digest, preprocessing version, teacher,
and evaluation family. Training rejects changed or reused shard identities.

## Shard Rows

Comment headers shown below are optional.

### Text

Format: `text_lines`

```text
one UTF-8 training record per line
```

### Dialogue

Format: `tsv`

```text
# prompt	response	optional_grounding
```

### Instruction

Format: `tsv`

```text
# task	domain	prompt	rationale	response	optional_quality
```

`quality` is a non-negative decimal and defaults to `1.0`.

### Preference

Format: `tsv`

```text
# prompt	chosen	rejected	optional_feedback	optional_weight
```

`weight` is a non-negative decimal and defaults to `1.0`.

### Vision

Format: `vision_tsv`

```text
# image_path	image_sha256	caption
```

The image path is resolved relative to the vision shard. Audited H100/H200
training requires the decoded media file SHA-256 field.

### Video

Format: `video_frames_tsv`

```text
# sequence_id	frame_index	frames_per_second	frame_path	frame_sha256	prompt	frame_caption
```

Each sequence needs at least two rows. Frame indices must be unique and
contiguous from zero, and all rows for a sequence must use one prompt and frame
rate.

### Tools

Format: `tsv`

```text
# request	tool_name
```

### Facts

Format: `tsv`

```text
# subject	relation	object	optional_confidence
```

`confidence` defaults to `1.0`.

### Rules

Format: `tsv`

```text
# name	premises	conclusion	optional_confidence
```

Each relational pattern is `subject,relation,object`. Separate multiple
premises with semicolons:

```text
grandparent composition	?x,parent,?y;?y,parent,?z	?x,grandparent,?z	0.98
```

## Frozen Evaluation

Development and evaluation shards use the same formats. They are audited for
exact, near-duplicate, solution-template, source-family, and perceptual media
contamination against training. Binary evaluation artifacts are accepted only
outside the training split.

See the hardware-specific directories in `data_templates/` for complete ledger
examples and `docs/H200_CAMPAIGN.md` for the exact-token H200 contract.
