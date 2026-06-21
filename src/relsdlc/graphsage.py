"""Training-free, inductive graph aggregation over the typed SDLC graph (numpy).

This is the first, deliberately cheap probe for Track B's question: **does graph
structure add signal beyond pairwise text cosine?** It uses *no learning* — there
are no weights to fit. Given pretrained node feature vectors (a frozen embedder's
output) and a typed adjacency built from the ``fixes`` and ``modifies`` relations,
it computes augmented node embeddings by 1–2 hops of mean aggregation:

    aug(v) = normalize( alpha * own(v) + (1 - alpha) * mean_r( mean(neighbours_r) ) )

where the inner mean is over neighbours reached by each *typed* edge role and the
outer mean is over the roles that node participates in. A node with no neighbours
keeps its own (normalized) feature; a node with no own feature (file/test nodes
have no text embedding) is *defined* purely by its neighbours — that is exactly the
structural signal cosine cannot see.

It is a GraphSAGE-style **mean aggregator** with the (untrained) projection fixed
to the identity, so the only thing it tests is whether neighbourhood averaging on
frozen features helps. A genuine *learned* GNN (R-GCN / GraphSAGE with trained
weights, torch) is the natural follow-up if this probe shows a signal; if it does
not, graph structure is not load-bearing at pilot scale on frozen features and the
learned variant is what to try next. The result is honest either way.

Determinism: every aggregation iterates neighbours in sorted-id order and uses
plain numpy means, so the augmented vectors are a pure function of the inputs.

Typed roles (a node aggregates from these edge roles only):

- ``fixes``     : PR  --fixes-->  issue      (and the reverse, issue <- PR)
- ``modifies``  : PR  --modifies--> file/test (and the reverse, file/test <- PR)

LEAKAGE GUARD. When the eval target is the ``fixes`` link itself (the
``issue_to_fixing_pr`` task), aggregating an issue *via its gold fixing PR* — or a
PR *via the gold issue it fixes* — would leak the answer through the graph. Pass
``exclude_fixes_pairs`` with the gold ``(issue_id, pr_id)`` pairs to drop exactly
those edges from the typed adjacency; the PR is still enriched by the files it
modifies, and the issue is reached only through structure that does not encode the
gold answer. For the ``diff_to_affected_test`` task the eval target is a
``modifies`` (PR→test) edge, so pass those gold pairs via ``exclude_modifies_pairs``
to keep that aggregation honest too.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Edge roles kept in the typed adjacency.
FIXES = "fixes"
MODIFIES = "modifies"


def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def build_typed_adjacency(
    fixes_edges,
    modifies_edges,
    *,
    exclude_fixes_pairs=None,
    exclude_modifies_pairs=None,
):
    """Build a symmetric, role-typed adjacency from ``fixes`` + ``modifies`` edges.

    ``fixes_edges``     : iterable of (pr_id, issue_id).
    ``modifies_edges``  : iterable of (pr_id, file_or_test_id).
    ``exclude_*_pairs`` : iterable of edge pairs to OMIT (leakage guard). For
        ``fixes`` the pair is (pr_id, issue_id); for ``modifies`` it is
        (pr_id, file_or_test_id). Order-insensitive.

    Returns ``adj``: ``dict[node_id -> dict[role -> set(neighbour_ids)]]``. Edges
    are stored in both directions under the same role, so every node can aggregate
    over the neighbours reachable by each role it participates in.
    """
    exclude_fixes = {frozenset(p) for p in (exclude_fixes_pairs or ())}
    exclude_mods = {frozenset(p) for p in (exclude_modifies_pairs or ())}

    adj: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    def add(a, b, role):
        adj[a][role].add(b)
        adj[b][role].add(a)

    for pr, iss in fixes_edges:
        if frozenset((pr, iss)) in exclude_fixes:
            continue
        add(pr, iss, FIXES)
    for pr, tgt in modifies_edges:
        if frozenset((pr, tgt)) in exclude_mods:
            continue
        add(pr, tgt, MODIFIES)

    # Freeze into plain dicts for deterministic iteration.
    return {n: {r: set(nb) for r, nb in roles.items()} for n, roles in adj.items()}


def _aggregate_once(node_id, base_vecs, prev_vecs, adj, dim):
    """One mean-aggregation hop for a single node, typed by edge role.

    ``base_vecs`` are the node's OWN features (frozen embedder; absent for
    file/test nodes). ``prev_vecs`` are the features to read from neighbours at this
    hop (the previous layer's output). Returns the un-normalized aggregate, or
    ``None`` if the node has neither an own feature nor any usable neighbour.
    """
    own = base_vecs.get(node_id)
    role_means = []
    for role in sorted(adj.get(node_id, {})):  # deterministic role order
        nbr_vecs = [prev_vecs[n] for n in sorted(adj[node_id][role]) if n in prev_vecs]
        if nbr_vecs:
            role_means.append(np.mean(np.stack(nbr_vecs), axis=0))
    if not role_means:
        # No neighbour features available: fall back to own (if any).
        return np.asarray(own, dtype=np.float64) if own is not None else None
    nbr_agg = np.mean(np.stack(role_means), axis=0)  # mean over roles
    if own is None:
        return nbr_agg  # file/test node: defined purely by structure
    return np.asarray(own, dtype=np.float64)  # combined below


def graphsage_aggregate(node_vecs, adj, *, alpha=0.5, hops=1, all_node_ids=None):
    """Augment node features by ``hops`` of typed mean aggregation (training-free).

    ``node_vecs``     : dict ``id -> np.ndarray`` of OWN (frozen) features. File/test
        nodes are typically absent here — they get a feature only from structure.
    ``adj``           : typed adjacency from :func:`build_typed_adjacency`.
    ``alpha``         : self-weight in ``alpha*own + (1-alpha)*neighbour_mean``.
    ``hops``          : 1 or 2 hops of aggregation.
    ``all_node_ids``  : optional EXTRA ids to guarantee in the output (e.g. eval
        candidates that have no own vector). It is always UNIONED with the
        ``node_vecs`` keys and every adjacency node — it never restricts the
        universe, because intermediate nodes (e.g. file nodes between a PR and the
        PRs that co-modify a file) must be materialized for aggregation to flow.

    Returns a dict ``id -> unit np.ndarray`` for every materialized node that ends
    up with a feature. Deterministic.
    """
    dim = None
    for v in node_vecs.values():
        dim = int(np.asarray(v).shape[-1])
        break
    if dim is None:
        return {}

    ids = set(node_vecs) | set(adj)
    for roles in adj.values():
        for nb in roles.values():
            ids |= nb
    if all_node_ids is not None:
        ids |= set(all_node_ids)
    ids = sorted(ids)

    base = {k: np.asarray(v, dtype=np.float64) for k, v in node_vecs.items()}
    # Layer-0 features: own vector if present, else the structural mean of
    # whatever neighbours already have a feature (lets file/test nodes bootstrap).
    cur: dict[str, np.ndarray] = {}
    for nid in ids:
        if nid in base:
            cur[nid] = base[nid]
    for nid in ids:
        if nid in cur:
            continue
        seed = _aggregate_once(nid, base, cur, adj, dim)
        if seed is not None:
            cur[nid] = seed

    for _ in range(hops):
        nxt: dict[str, np.ndarray] = {}
        for nid in ids:
            own = base.get(nid)
            role_means = []
            for role in sorted(adj.get(nid, {})):
                nbr = [cur[n] for n in sorted(adj[nid][role]) if n in cur]
                if nbr:
                    role_means.append(np.mean(np.stack(nbr), axis=0))
            if role_means:
                nbr_agg = np.mean(np.stack(role_means), axis=0)
            else:
                nbr_agg = None
            if own is not None and nbr_agg is not None:
                mixed = alpha * own + (1.0 - alpha) * nbr_agg
            elif own is not None:
                mixed = own
            elif nbr_agg is not None:
                mixed = nbr_agg
            elif nid in cur:
                mixed = cur[nid]
            else:
                continue
            nxt[nid] = mixed
        cur = nxt

    return {nid: _unit(v) for nid, v in cur.items()}


def augmented_vecs(
    node_vecs,
    fixes_edges,
    modifies_edges,
    *,
    alpha=0.5,
    hops=1,
    exclude_fixes_pairs=None,
    exclude_modifies_pairs=None,
    all_node_ids=None,
):
    """Convenience: build the typed adjacency and return augmented unit vectors.

    The returned dict can be scored directly with
    :func:`relsdlc.tower.run_cosine_on_vecs` / ``relsdlc.tower._eval`` on a
    benchmark whose candidates resolve to these node ids.
    """
    adj = build_typed_adjacency(
        fixes_edges,
        modifies_edges,
        exclude_fixes_pairs=exclude_fixes_pairs,
        exclude_modifies_pairs=exclude_modifies_pairs,
    )
    return graphsage_aggregate(
        node_vecs, adj, alpha=alpha, hops=hops, all_node_ids=all_node_ids
    )
