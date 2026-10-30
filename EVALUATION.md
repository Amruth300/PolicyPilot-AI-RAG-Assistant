# Starter RAG evaluation plan

Before claiming that ComplyAI is accurate, create a small test set from documents you are allowed to use.

For each test case, record:
- question
- expected supporting document and page
- expected answer points
- whether the retrieved top-k passages contain the evidence
- whether the generated answer is supported by those passages
- whether the answer correctly abstains when evidence is absent

## Metrics to learn

**Recall@K (retrieval):** the proportion of questions for which at least one relevant passage appears in the top K retrieved results.

**Answer relevance:** whether the answer addresses the question.

**Faithfulness / groundedness:** whether claims in the answer are supported by retrieved passages.

Do not report a performance percentage until you have run a documented test set and retained the results. Similarity scores are not answer-accuracy scores.
