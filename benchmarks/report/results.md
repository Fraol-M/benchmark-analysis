# Benchmark Report — NL → AtomSpace Extraction

Generated: 2026-05-01 16:57:27

---

## Summary

| Metric | demo |
| --- | --- |
| Total Cases | 37 |
| Keyword Hits | 36/37 |
| **Extraction Accuracy** | **97.3%** |
| Avg Keyword Precision | 0.94 |
| Avg Atom Count | 3.5 |
| Avg Unique Heads | 3.0 |
| Avg Rule Ratio | 0.20 |
| Avg Atoms / Sentence | 2.28 |
| Cases with Rules | 19 |
| Cases with Facts | 36 |
| Avg Ingest Latency | 9.38s |
| Total Errors | 1 |

## Extraction Accuracy by Category

| Category | demo |
| --- | --- |
| fact-extraction | 100% | 
| inheritance | 100% | 
| multi-premise-rule | 100% | 
| multi-variable-rule | 100% | 
| negation | 100% | 
| negative | 100% | 
| open-question | 100% | 
| paragraph | 86% | 
| paraphrase | 100% | 
| single-hop-rule | 100% | 
| transitive-chain | 100% | 

## Avg Atom Count by Category

| Category | demo |
| --- | --- |
| fact-extraction | 2.5 | 
| inheritance | 2.0 | 
| multi-premise-rule | 4.0 | 
| multi-variable-rule | 3.0 | 
| negation | 3.0 | 
| negative | 2.0 | 
| open-question | 2.5 | 
| paragraph | 8.7 | 
| paraphrase | 1.0 | 
| single-hop-rule | 2.1 | 
| transitive-chain | 3.5 | 

## Accuracy by Input Complexity (Hop Depth)

| Hop Depth | demo |
| --- | --- |
| 0 | 100% | 
| 1 | 100% | 
| 2 | 80% | 
| 3 | 100% | 

## Detailed Results — demo

| Case | Category | Hops | ✓ | KW | Atoms | Heads | Rules | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wings-fly-rule | single-hop-rule | 1 | ✅ | 4/4 | 3 | 3 | 1 | 14.6s |
| programmer-solves | single-hop-rule | 1 | ✅ | 4/4 | 2 | 2 | 1 | 7.6s |
| no-proof-negative | negative | 0 | ✅ | 3/3 | 2 | 2 | 0 | 3.7s |
| para-hospital-staff | paragraph | 1 | ✅ | 9/11 | 8 | 6 | 1 | 20.6s |
| para-supply-chain | paragraph | 2 | ✅ | 11/11 | 14 | 11 | 4 | 20.5s |
| para-ecology | paragraph | 3 | ✅ | 8/10 | 9 | 7 | 3 | 27.6s |
| para-legal-contract | paragraph | 2 | ✅ | 12/12 | 11 | 11 | 1 | 32.8s |
| para-research-lab | paragraph | 1 | ✅ | 12/12 | 9 | 7 | 0 | 27.9s |
| para-city-infrastructure | paragraph | 2 | ❌ | 0/11 | 0 | 0 | 0 | 13.3s |
| para-mixed-short-long | paragraph | 1 | ✅ | 11/11 | 10 | 9 | 2 | 22.3s |
| fish-smart-yesno | single-hop-rule | 1 | ✅ | 3/3 | 2 | 2 | 1 | 6.5s |
| dog-animal-yesno | inheritance | 1 | ✅ | 3/3 | 2 | 2 | 0 | 6.3s |
| human-mortal-yesno | single-hop-rule | 1 | ✅ | 2/3 | 2 | 2 | 0 | 7.6s |
| coffee-awake-yesno | single-hop-rule | 1 | ✅ | 3/3 | 2 | 2 | 1 | 6.1s |
| parent-caring-yesno | single-hop-rule | 1 | ✅ | 3/3 | 2 | 2 | 1 | 9.7s |
| teacher-educates-yesno | single-hop-rule | 1 | ✅ | 3/3 | 2 | 2 | 1 | 7.3s |
| doctor-heals-yesno | single-hop-rule | 1 | ✅ | 3/3 | 2 | 2 | 1 | 5.8s |
| frog-basic-facts | fact-extraction | 0 | ✅ | 4/4 | 3 | 3 | 0 | 5.8s |
| frog-green-rule | multi-premise-rule | 1 | ✅ | 4/4 | 4 | 4 | 1 | 7.2s |
| negation-basic | negation | 0 | ✅ | 4/4 | 3 | 2 | 0 | 5.4s |
| type-declaration | fact-extraction | 0 | ✅ | 4/4 | 2 | 1 | 0 | 5.6s |
| property-color | fact-extraction | 0 | ✅ | 4/4 | 2 | 1 | 0 | 5.9s |
| inheritance-chain | inheritance | 2 | ✅ | 3/3 | 2 | 1 | 0 | 6.0s |
| student-expert-rule | multi-variable-rule | 1 | ✅ | 5/5 | 3 | 3 | 1 | 9.3s |
| spatial-location | multi-variable-rule | 1 | ✅ | 3/3 | 3 | 3 | 1 | 7.2s |
| transitive-chain-2hop | transitive-chain | 2 | ✅ | 4/4 | 3 | 2 | 2 | 5.3s |
| freezing-chain-3hop | transitive-chain | 3 | ✅ | 3/4 | 4 | 2 | 3 | 7.9s |
| smart-open-who | open-question | 1 | ✅ | 5/5 | 3 | 2 | 1 | 5.8s |
| eat-open-what | open-question | 0 | ✅ | 4/4 | 2 | 1 | 0 | 3.5s |
| multiple-facts | fact-extraction | 0 | ✅ | 4/4 | 3 | 3 | 0 | 3.2s |
| partof-rule | multi-premise-rule | 1 | ✅ | 5/5 | 4 | 4 | 1 | 7.4s |
| paraphrase-eat-1 | paraphrase | 0 | ✅ | 2/3 | 1 | 1 | 0 | 4.5s |
| paraphrase-eat-2 | paraphrase | 0 | ✅ | 2/2 | 1 | 1 | 0 | 2.8s |
| paraphrase-eat-3 | paraphrase | 0 | ✅ | 3/3 | 1 | 1 | 0 | 3.5s |
| paraphrase-class-1 | paraphrase | 0 | ✅ | 2/2 | 1 | 1 | 0 | 3.7s |
| paraphrase-class-2 | paraphrase | 0 | ✅ | 2/2 | 1 | 1 | 0 | 3.1s |
| paraphrase-class-3 | paraphrase | 0 | ✅ | 2/2 | 1 | 1 | 0 | 4.2s |

### Errors — demo

- **para-city-infrastructure**: Gemini API error: Server disconnected without sending a response.