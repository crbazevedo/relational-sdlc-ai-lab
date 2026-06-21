"""Procedural generator for a relational SDLC benchmark with a learnable structure.

The point of this dataset is to be *hard for vanilla text retrieval and solvable
by relation supervision* — a controlled stand-in for the real phenomenon the lab
studies. It is synthetic and labeled exploratory; it demonstrates the mechanism,
not a real-world result.

Design
------
Each artifact belongs to a latent **component** (a module of a codebase). The
vocabulary splits into:

- **impl tokens** — component-specific (e.g. ``impl_c3_07``). They identify the
  component but are *rare* in issue text.
- **topic tokens** — shared symptom words (e.g. ``topic_19``), spread across many
  components, so they are ambiguous.

An **issue** is topic-heavy with only a weak sprinkle of its component's impl
tokens. Its **fixing PR** is impl-heavy. So surface overlap between an issue and
its true fix is small, while *hard-negative* PRs from other components that happen
to share topic words look more similar on the surface. Vanilla cosine is fooled;
a model trained on the ``fixes`` relation learns that impl tokens predict the fix
and topic tokens do not, and recovers the true link.

``generate`` is fully deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Artifact:
    id: str
    type: str  # "issue" | "pull_request"
    component: int
    tokens: list[str]
    split: str  # "train" | "test"

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass
class Query:
    query_id: str
    query_record: str
    candidates: list[str]
    relevant: list[str]
    hard_negatives: list[str]
    split: str


@dataclass
class SynthDataset:
    artifacts: list[Artifact]
    fixes: list[tuple[str, str]]  # (pr_id, issue_id)
    queries: list[Query]
    params: dict = field(default_factory=dict)

    def by_id(self) -> dict[str, Artifact]:
        return {a.id: a for a in self.artifacts}


def generate(
    seed: int = 7,
    n_components: int = 10,
    issues_per_component: int = 24,
    impl_per_component: int = 12,
    n_topics: int = 40,
    issue_topic_tokens: int = 10,
    issue_impl_tokens: int = 2,
    pr_impl_tokens: int = 8,
    pr_topic_tokens: int = 6,
    n_hard_negatives: int = 5,
    n_random_negatives: int = 6,
    train_frac: float = 0.6,
) -> SynthDataset:
    rng = np.random.default_rng(seed)

    # Tokens are underscore-free single words so the shared tokenizer keeps them
    # intact (it splits on underscores) and impl/topic namespaces never collide.
    # impl tokens are component-specific and RARE (one component only) -> highly
    # predictive of the fix. topic tokens are GLOBAL symptom words -> common and
    # ambiguous. Component identity lives only in impl tokens.
    impl_tokens = {
        c: [f"implc{c}n{j:02d}" for j in range(impl_per_component)]
        for c in range(n_components)
    }
    topics = [f"topic{t:02d}" for t in range(n_topics)]

    def pick(pool: list, k: int) -> list:
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]

    artifacts: list[Artifact] = []
    fixes: list[tuple[str, str]] = []

    counter = 0
    for c in range(n_components):
        for _ in range(issues_per_component):
            counter += 1
            split = "train" if rng.random() < train_frac else "test"
            # 'syn' namespace keeps these ids globally unique alongside other datasets.
            issue_id = f"issue:syn{counter}"
            pr_id = f"pr:syn{counter}"

            # Issue: a couple of (rare) impl tokens buried under many (common)
            # symptom topics.
            issue_topics = pick(topics, issue_topic_tokens)
            issue_tokens = issue_topics + pick(impl_tokens[c], issue_impl_tokens)

            # True PR: impl-heavy with topics sampled globally. Its only RELIABLE
            # surface link to the issue is the rare impl token; topic overlap with
            # the issue is incidental, so vanilla cosine — which weights common
            # topics equally — is often misled by topic-sharing distractors.
            pr_tokens = pick(impl_tokens[c], pr_impl_tokens) + pick(topics, pr_topic_tokens)

            artifacts.append(Artifact(issue_id, "issue", c, issue_tokens, split))
            artifacts.append(Artifact(pr_id, "pull_request", c, pr_tokens, split))
            fixes.append((pr_id, issue_id))

    all_prs = [a for a in artifacts if a.type == "pull_request"]
    fix_pr_of_issue = {iss: pr for pr, iss in fixes}

    def topic_overlap(a: Artifact, b: Artifact) -> int:
        return len(set(t for t in a.tokens if t.startswith("topic"))
                   & set(t for t in b.tokens if t.startswith("topic")))

    queries: list[Query] = []
    for issue in (a for a in artifacts if a.type == "issue"):
        true_pr = fix_pr_of_issue[issue.id]
        others = [pr for pr in all_prs if pr.component != issue.component]
        # Hard negatives: other-component PRs with the most surface (topic) overlap.
        others_sorted = sorted(others, key=lambda pr: (-topic_overlap(issue, pr), pr.id))
        hard = [pr.id for pr in others_sorted[:n_hard_negatives]]
        remaining = [pr.id for pr in others_sorted[n_hard_negatives:]]
        rnd = pick(remaining, n_random_negatives) if remaining else []
        candidates = [true_pr] + hard + rnd
        # Deterministic shuffle of the candidate order.
        order = rng.permutation(len(candidates))
        candidates = [candidates[i] for i in order]
        queries.append(Query(
            query_id=f"q-{issue.id}",
            query_record=issue.id,
            candidates=candidates,
            relevant=[true_pr],
            hard_negatives=hard,
            split=issue.split,
        ))

    return SynthDataset(
        artifacts=artifacts,
        fixes=fixes,
        queries=queries,
        params={
            "seed": seed,
            "n_components": n_components,
            "issues_per_component": issues_per_component,
            "impl_per_component": impl_per_component,
            "n_topics": n_topics,
            "issue_topic_tokens": issue_topic_tokens,
            "issue_impl_tokens": issue_impl_tokens,
            "pr_impl_tokens": pr_impl_tokens,
            "pr_topic_tokens": pr_topic_tokens,
            "n_hard_negatives": n_hard_negatives,
            "n_random_negatives": n_random_negatives,
            "train_frac": train_frac,
        },
    )
