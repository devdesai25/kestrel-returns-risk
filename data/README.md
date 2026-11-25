# Data directory

Place the eight files from the Kestrel returns data pack here:

- `train.csv`
- `test_unlabelled.csv`
- `customers.csv`
- `products.csv`
- `sample_submission.csv`
- `README.txt`
- `email-thread.txt`
- `ops-policy.pdf`

Or point `KESTREL_DATA_DIR` at a directory that already contains them.

These files are **git-ignored** on purpose. Ops policy §10 forbids publishing customer /
operational data to public repositories. Do not commit CSVs, the PDF, or the email thread.
