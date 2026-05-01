# Benchmark Report — NL → AtomSpace Extraction

Generated: 2026-05-01 16:07:11

---

## Summary

| Metric | mock |
| --- | --- |
| Total Cases | 3 |
| Keyword Hits | 2/3 |
| **Extraction Accuracy** | **66.7%** |
| Avg Keyword Precision | 0.75 |
| Avg Atom Count | 4.0 |
| Avg Unique Heads | 2.5 |
| Avg Rule Ratio | 0.25 |
| Avg Atoms / Sentence | 2.00 |
| Cases with Rules | 2 |
| Cases with Facts | 3 |
| Avg Ingest Latency | 1.20s |
| Total Errors | 0 |

## Extraction Accuracy by Category

| Category | mock |
| --- | --- |
| fact-extraction | 100% | 
| paragraph | 50% | 

## Avg Atom Count by Category

| Category | mock |
| --- | --- |
| fact-extraction | 3.0 | 
| paragraph | 8.0 | 

## Accuracy by Input Complexity (Hop Depth)

| Hop Depth | mock |
| --- | --- |
| 0 | 100% | 
| 1 | 50% | 

## Detailed Results — mock

| Case | Category | Hops | ✓ | KW | Atoms | Heads | Rules | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test-1 | fact-extraction | 0 | ✅ | 2/2 | 3 | 2 | 0 | 1.1s |