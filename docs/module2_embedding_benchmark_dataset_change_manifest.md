# Module 2 benchmark dataset authoring boundary

## Outcome

The embedding benchmark now requires reviewed, content-addressed authoring
evidence. A benchmark dataset can no longer omit who reviewed its gold labels
or which exact source catalogs produced its documents.

## Added

- safe document- and gold-case-catalog contracts;
- an immutable dataset compiler and readiness report;
- least-privilege pgvector export of safe trait fields;
- atomic command-line tools for safe catalog export and dataset compilation;
- constraint, hard-negative, unsupported-location, multilingual, duplicate,
  privacy, rights, lineage, and coverage validation;
- schema-only document and case catalog examples;
- focused regression tests.

## Corrected

Geographic accuracy now measures supported requests only. Unsupported-location
requests remain measured by the separate false-match rate, so a correct
abstention is not counted as a geographic error.

## Unchanged boundaries

- no raw identifiers or embeddings are exported into authoring catalogs;
- no historical audience is activation-eligible;
- no model is selected or approved by this change;
- no production features are built;
- no audience activation or destination export is performed.
