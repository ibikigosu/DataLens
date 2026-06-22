# Local data layout

The data directories are present in Git, but downloaded and derived records are ignored.
This prevents public procurement extracts and generated controlled-defect datasets from bloating the teaching repository.

## Directories

- `raw/` contains immutable USAspending CSV extracts.
- `processed/` contains normalized vendor and transaction Parquet files.
- `external/` is reserved for future documented enrichment sources.
- `manifests/` contains small provenance records that may be versioned.

Never edit a raw source extract in place.
Run the acquisition and preparation modules to reproduce local data.
