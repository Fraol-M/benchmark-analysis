# Benchmark Report — Demo vs PLN-RAG

Generated: 2026-04-29 10:43:15

---

## Summary

| Metric | demo |
| --- | --- |
| Total Cases | 1 |
| Correct | 0/1 |
| **Accuracy** | **0.0%** |
| Avg Keyword Precision | 0.00 |
| Avg Atom Count | 0.0 |
| Cases with Rules | 0 |
| Cases with Facts | 0 |
| Avg Ingest Latency | 8.15s |
| Avg Query Latency | 0.00s |
| Total Errors | 1 |

## Accuracy by Category

| Category | demo |
| --- | --- |
| single-hop-rule | 0% | 

## Accuracy by Reasoning Depth

| Hop Depth | demo |
| --- | --- |
| 1 | 0% | 

## Detailed Results — demo

| Case | Category | Hops | Correct | Answer (truncated) | Atoms |
| --- | --- | --- | --- | --- | --- |
| fish-smart-yesno | single-hop-rule | 1 | ❌ |  | 0 |

### Errors — demo

- **fish-smart-yesno**: ingest=Gemini API error: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'API key expired. Please renew the API key.', 'status': 'INVALID_ARGUMENT', 'details': [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'API_KEY_INVALID', 'domain': 'googleapis.com', 'metadata': {'service': 'generativelanguage.googleapis.com'}}, {'@type': 'type.googleapis.com/google.rpc.LocalizedMessage', 'locale': 'en-US', 'message': 'API key expired. Please renew the API key.'}]}}, query=None
