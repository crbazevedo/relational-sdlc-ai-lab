# Position Paper Outline

Working title:

> Relational-Geometric Learning for SDLC and Software Agents

Core claim:

> SLMs and embeddings trained jointly over relational SDLC graphs can produce
> more reliable software engineering systems than SLMs with generic text RAG.

Contribution:

- not a single new loss;
- a unified architecture for relation-trained embeddings, SLM policy heads,
  graph retrieval, and verifiable SDLC feedback;
- public datasets and benchmarks.

Initial falsifiable hypothesis:

> Relation-trained embeddings improve file/test retrieval Recall@K and MRR over
> off-the-shelf embeddings on frozen public GitHub SDLC relation tasks.
