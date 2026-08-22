# Part 3 Retrieval Evaluation

Evaluation is performed at document level.
Top-3 retrieved chunks are mapped to their parent document IDs and deduplicated before scoring.

## Query 1
**Query:** What is the return policy for apparel?
**Gold document:** POL001
**Retrieved document IDs (top-3, deduplicated):** ['POL001', 'POL010']
**Relevant retrieved documents:** 1
**Precision@3:** 1/3 = 0.3333
**Recall@3:** 1/1 = 1.0000

## Query 2
**Query:** What is the return policy for footwear?
**Gold document:** POL002
**Retrieved document IDs (top-3, deduplicated):** ['POL002', 'POL010', 'POL001']
**Relevant retrieved documents:** 1
**Precision@3:** 1/3 = 0.3333
**Recall@3:** 1/1 = 1.0000

## Query 3
**Query:** What is the return policy for electronics?
**Gold document:** POL003
**Retrieved document IDs (top-3, deduplicated):** ['POL010', 'POL003']
**Relevant retrieved documents:** 1
**Precision@3:** 1/3 = 0.3333
**Recall@3:** 1/1 = 1.0000

## Query 4
**Query:** When is a COD refund processed?
**Gold document:** POL005
**Retrieved document IDs (top-3, deduplicated):** ['POL005', 'POL006']
**Relevant retrieved documents:** 1
**Precision@3:** 1/3 = 0.3333
**Recall@3:** 1/1 = 1.0000

## Query 5
**Query:** What is the exchange policy?
**Gold document:** POL013
**Retrieved document IDs (top-3, deduplicated):** ['POL013', 'POL010']
**Relevant retrieved documents:** 1
**Precision@3:** 1/3 = 0.3333
**Recall@3:** 1/1 = 1.0000

## Final averages
**Average Precision@3:** (0.3333 + 0.3333 + 0.3333 + 0.3333 + 0.3333) / 5 = 0.3333
**Average Recall@3:** (1.0000 + 1.0000 + 1.0000 + 1.0000 + 1.0000) / 5 = 1.0000
