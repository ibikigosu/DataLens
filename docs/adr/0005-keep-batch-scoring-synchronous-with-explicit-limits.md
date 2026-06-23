# Keep batch scoring synchronous with explicit limits

Batch scoring will remain synchronous for files up to 10 MB each and 50,000 combined rows.
A completed request returns a stable run identifier and summary, while findings are retrieved separately.
This keeps the demonstration reliable and understandable while defining the boundary at which a future deployment would introduce asynchronous workers.
