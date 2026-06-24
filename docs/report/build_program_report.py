#!/usr/bin/env python3
"""Build the comprehensive program report PDF (state through R16E).

Pure reportlab/Platypus; registers Arial Unicode so Greek/math/arrow glyphs
render (built-in PDF fonts would show black boxes). Run with the venv that has
reportlab installed:

    .venv-np/bin/python docs/report/build_program_report.py
"""
from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable, ListFlowable, ListItem,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "relational-sdlc-lab-report-2026-06-23.pdf")

# ---------------------------------------------------------------- fonts
SUP = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Body", f"{SUP}/Arial Unicode.ttf"))      # full glyph coverage
pdfmetrics.registerFont(TTFont("Body-B", f"{SUP}/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-I", f"{SUP}/Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("Body-BI", f"{SUP}/Arial Bold Italic.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B",
                              italic="Body-I", boldItalic="Body-BI")

# ---------------------------------------------------------------- palette
NAVY = colors.HexColor("#16314f")
ACCENT = colors.HexColor("#2f5d8a")
LIGHT = colors.HexColor("#eef2f7")
ROWALT = colors.HexColor("#f6f8fb")
RULE = colors.HexColor("#c9d3e0")
GRAY = colors.HexColor("#5a6573")
GOOD = colors.HexColor("#1f7a4d")
BAD = colors.HexColor("#b03030")

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()

def style(name, **kw):
    base = dict(fontName="Body", fontSize=9.6, leading=13.6, textColor=colors.HexColor("#1c2530"))
    base.update(kw)
    return ParagraphStyle(name, **base)

BODY = style("body", alignment=TA_JUSTIFY, spaceAfter=6)
BODY_T = style("bodytight", alignment=TA_JUSTIFY, spaceAfter=2)
LEAD = style("lead", fontSize=10.6, leading=15.2, alignment=TA_JUSTIFY,
             textColor=colors.HexColor("#2b3645"), spaceAfter=7)
H1 = style("h1", fontName="Body-B", fontSize=15, leading=18, textColor=NAVY,
           spaceBefore=15, spaceAfter=3, keepWithNext=1)
H2 = style("h2", fontName="Body-B", fontSize=11.4, leading=15, textColor=ACCENT,
           spaceBefore=9, spaceAfter=2, keepWithNext=1)
CAP = style("cap", fontName="Body-I", fontSize=8.2, leading=10.5, textColor=GRAY,
            spaceBefore=2, spaceAfter=9)
SMALL = style("small", fontSize=8.6, leading=11.6, textColor=GRAY)
BULLET = style("bullet", alignment=TA_LEFT, fontSize=9.4, leading=13.0, spaceAfter=2)
TH = style("th", fontName="Body-B", fontSize=8.4, leading=10.4, textColor=colors.white)
TH_L = style("thl", fontName="Body-B", fontSize=8.4, leading=10.4, textColor=colors.white, alignment=TA_LEFT)
TC = style("tc", fontSize=8.4, leading=10.6)
TC_C = style("tcc", fontSize=8.4, leading=10.6, alignment=TA_CENTER)
TC_B = style("tcb", fontName="Body-B", fontSize=8.4, leading=10.6)
TITLE = style("title", fontName="Body-B", fontSize=21, leading=24, textColor=NAVY)
SUBTITLE = style("subtitle", fontName="Body", fontSize=12, leading=16, textColor=ACCENT)
META = style("meta", fontSize=9, leading=12.5, textColor=GRAY)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items, st=BULLET, bullet="•"):
    return ListFlowable(
        [ListItem(P(x, st), leftIndent=12, value=bullet) for x in items],
        bulletType="bullet", start=bullet, leftIndent=10, bulletFontName="Body",
        bulletFontSize=8, spaceBefore=1, spaceAfter=5,
    )


def heading_rule():
    return HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=1, spaceAfter=6)


def table(data, col_w, header=True, align_cols=None, font=8.4, lead=10.6):
    """data: list of rows; cells are strings (wrapped as Paragraphs) or flowables."""
    align_cols = align_cols or {}
    rows = []
    for r, row in enumerate(data):
        out = []
        for c, cell in enumerate(row):
            if isinstance(cell, (Paragraph, Table)):
                out.append(cell)
            else:
                if header and r == 0:
                    st = TH_L if c == 0 else TH
                    out.append(Paragraph(str(cell), st))
                else:
                    a = align_cols.get(c, TA_LEFT)
                    st = ParagraphStyle(f"c{r}{c}", parent=TC, alignment=a)
                    out.append(Paragraph(str(cell), st))
        rows.append(out)
    t = Table(rows, colWidths=col_w, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, NAVY),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROWALT]),
        ]
    else:
        cmds += [("LINEABOVE", (0, 0), (-1, 0), 0.8, NAVY)]
    t.setStyle(TableStyle(cmds))
    return t


# ---------------------------------------------------------------- page furniture
def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # footer rule + text
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(2.0 * cm, 1.35 * cm, w - 2.0 * cm, 1.35 * cm)
    canvas.setFont("Body", 7.6)
    canvas.setFillColor(GRAY)
    canvas.drawString(2.0 * cm, 1.0 * cm,
                      "Relational SDLC AI Lab — program report (exploratory, pilot-scale; experiment cards are the source of truth)")
    canvas.drawRightString(w - 2.0 * cm, 1.0 * cm, "p. %d" % doc.page)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.7 * cm,
        title="Relational-Geometric Learning over the SDLC — Program Report",
        author="Relational SDLC AI Lab",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])

    E = []  # the story
    W = doc.width

    # ===================================================== title block
    E.append(Spacer(1, 4))
    E.append(P("Relational-Geometric Learning over the<br/>Software Development Lifecycle", TITLE))
    E.append(Spacer(1, 5))
    E.append(P("Program report — data, pipelines, methods, hypotheses, experiments, results, and next steps", SUBTITLE))
    E.append(Spacer(1, 8))
    E.append(HRFlowable(width="100%", thickness=1.4, color=NAVY, spaceAfter=6))
    E.append(P("Relational SDLC AI Lab &nbsp;·&nbsp; public research repository &nbsp;·&nbsp; compiled 2026-06-23 &nbsp;·&nbsp; state through wave R19", META))
    E.append(Spacer(1, 10))

    # ---- at-a-glance box
    glance = [
        [P("At a glance", TC_B), P("", TC)],
        ["Thesis", "Embeddings (and later SLMs) trained over <b>verifiable SDLC relations</b> generalize across repositories better than generic text similarity."],
        ["Headline win", "Relation-loss <b>LoRA inside the representation</b> beats the frozen embedder cross-repo, and the gain <b>grows with data</b>: ΔR@1 +0.07 (pilot) → +0.080 (55 repos) → +0.114 (78-repo dense Tier-2) → <b>+0.142</b> with same-repo <b>hard negatives</b> (R18/R19, first wave trained on the Apple M5 GPU)."],
        ["Robustness", "Positive on <b>all 5</b> held-out-repo splits (ΔR@1 +0.061±0.021); on the headline split both query- and repo-cluster <b>95% CIs exclude zero</b> (R17a). A thin parameter-free <b>graph lift</b> stacks on top (LoRA+graph R@1 0.690), robust across α and saturated at 1 hop."],
        ["Strongest negatives", "Learned heads on <i>frozen</i> vectors overfit at pilot scale; a code-MLM base collapses; <b>more text hurts</b> (de-truncation −0.09 to −0.15 R@1)."],
        ["Discipline", "CPU-first, one change vs. the embedder-cosine control, de-referenced cross-repo splits, every run emits an experiment card, honest negatives kept."],
    ]
    gt = Table(glance, colWidths=[3.1 * cm, W - 3.1 * cm])
    gt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("SPAN", (0, 0), (-1, 0)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
        ("FONTNAME", (0, 1), (0, -1), "Body-B"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.8),
        ("LEADING", (0, 1), (-1, -1), 12),
        ("TEXTCOLOR", (0, 1), (0, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]))
    # wrap value cells as paragraphs for proper wrapping
    glance_h = style("glanceh", fontName="Body-B", fontSize=9.8, textColor=colors.white)
    glance_wrapped = [[P("At a glance", glance_h), P("", TC)]]
    for label, val in glance[1:]:
        glance_wrapped.append([P(label, ParagraphStyle("gl", parent=TC, fontName="Body-B", textColor=NAVY, fontSize=8.8, leading=12)),
                               P(val, ParagraphStyle("gv", parent=TC, fontSize=8.8, leading=12))])
    gt = Table(glance_wrapped, colWidths=[3.1 * cm, W - 3.1 * cm])
    gt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]))
    E.append(gt)
    E.append(Spacer(1, 4))

    # ===================================================== 0. abstract
    E.append(P("Abstract", H1))
    E.append(heading_rule())
    E.append(P(
        "This report synthesizes the work of a public research lab studying "
        "<i>relational-geometric learning</i> over software-development-lifecycle (SDLC) "
        "records. The lab tests one claim: that learning over <b>verifiable relations</b> among "
        "artifacts — a pull request <i>fixes</i> an issue, a commit <i>modifies</i> a file, a test "
        "<i>covers</i> a symbol — yields retrieval that generalizes across repositories better "
        "than generic text similarity. We instantiate a typed artifact/relation graph, a ladder "
        "of relation operators (diagonal metric, low-rank two-tower, identity-initialized "
        "operator, a typed graph-aggregation lift, and a LoRA-fine-tuned encoder), and a frozen, "
        "leakage-guarded benchmark scored by Recall@K, MRR, and hard-negative accuracy. The arc "
        "is deliberately one-change-at-a-time against an <i>embedder-cosine</i> control. The "
        "settled findings: pretrained embeddings are the cross-repo substrate; the relational "
        "contribution must live <b>inside the representation</b> (LoRA), not in a head bolted onto "
        "frozen vectors; and the LoRA win is robust and grows with data density. We report the "
        "datasets, pipelines, methods, hypotheses, the full scope of experiments (waves R3–R16E), "
        "results, conclusions, and next steps.", LEAD))

    # ===================================================== 1. overview
    E.append(P("1&nbsp;&nbsp;Overview and Thesis", H1))
    E.append(heading_rule())
    E.append(P(
        "Modern software-engineering assistants are built largely on <i>text</i> retrieval: an "
        "embedding model maps an artifact to a vector and relevance is cosine similarity. That "
        "works when the answer restates the question, but SDLC artifacts are bound by <b>relations</b> "
        "that are often not surface-visible yet are <i>verifiable</i> — minable from closing "
        "keywords, diff contents, or coverage maps — which makes them an unusually trustworthy "
        "supervision signal.", BODY))
    E.append(P(
        "<b>Thesis.</b> Embeddings (and later small language models) trained over verifiable SDLC "
        "relations produce more reliable software-engineering systems than models that rely on "
        "generic text similarity.", BODY))
    E.append(P(
        "The first target is narrow on purpose: not an autonomous coding agent, but a <b>measurable "
        "relational retrieval layer</b> — issue → likely fixing PRs; diff → affected tests; failing "
        "log → likely files; PR → risk and missing-test candidates. This is MVP-1 of the program. "
        "The enduring deliverable is not any single model but a <b>frozen public benchmark with "
        "honest baselines and full provenance</b> that makes these questions falsifiable. Five "
        "standing goals govern the work: public research hygiene (G1), falsifiable retrieval gains "
        "(G2), dataset provenance (G3), relation-aware models (G4), and agentic readiness (G5).", BODY))

    # ===================================================== 2. data
    E.append(P("2&nbsp;&nbsp;Data and Datasets", H1))
    E.append(heading_rule())
    E.append(P(
        "Every dataset is a frozen snapshot of public, permissively-licensed GitHub repositories "
        "(plus one original synthetic fixture). Records are typed SDLC artifacts; edges are typed "
        "relations. Each record and edge carries full provenance — <font name='Body-B'>source_url</font>, "
        "<font name='Body-B'>retrieved_at</font>, <font name='Body-B'>license</font>, a real "
        "<font name='Body-B'>sha256</font> content hash, a transform lineage, an extraction "
        "<font name='Body-B'>method</font>, and a <font name='Body-B'>valid_from</font> timestamp used "
        "for temporal leakage control. JSON schemas for records, edges, benchmark queries, and the "
        "source/dataset/experiment cards live under <font name='Body-B'>schemas/</font>.", BODY))

    E.append(P("2.1&nbsp;&nbsp;Dataset tiers", H2))
    data_rows = [
        ["Dataset", "Repos", "Records", "fixes edges", "Bench queries", "q / repo", "Role"],
        ["<b>datebox</b> (synthetic)", "—", "original", "controlled", "94 held-out", "—", "Mechanism probe (CC0); timezone-bug worked example; runs offline from a clean checkout"],
        ["<b>pilot</b> (P1)", "20", "2,087", "356", "356", "~18", "First real snapshot; issue→PR; the workhorse for operator ablations"],
        ["<b>scale</b> (Tier-2 entry)", "55", "3,672", "562", "562", "~10", "Breadth check; bag-of-tokens + LoRA at larger repo count"],
        ["<b>tier2</b> (dense)", "78", "16,998", "2,744", "2,744", "~35", "Wide and ~3.5x denser per repo; the representation-learning regime"],
    ]
    ac = {1: TA_CENTER, 2: TA_CENTER, 3: TA_CENTER, 4: TA_CENTER, 5: TA_CENTER}
    E.append(table(data_rows, [3.0 * cm, 1.0 * cm, 1.5 * cm, 1.5 * cm, 1.9 * cm, 1.2 * cm, W - 12.1 * cm], align_cols=ac))
    E.append(P("Table 1. Frozen dataset tiers. Tier-2 records are 2,282 issue + 14,716 pull_request; "
               "the snapshot is pruned to records referenced by an edge or query. Prior snapshots are "
               "never mutated (other experiments depend on them) and ids are namespaced and disjoint "
               "(pinned by a test).", CAP))
    E.append(P(
        "Two further <i>derived</i> sets support specific waves: a <b>full-text</b> variant of the "
        "pilot that de-truncates issue/PR bodies (500 → 8000 chars) for the paired truncation control "
        "(R14), and a <b>deep-content</b> set of long file bodies for the diff → affected-test "
        "chunking study (R16A). The graph-learning track adds file-level structure to the pilot via a "
        "one-time live enrichment: <b>497 file + 239 test nodes</b> (736 total) and <b>1,356 "
        "<i>modifies</i> edges</b> over 18 repos, each edge observed directly from a diff with "
        "confidence 1.0.", BODY))

    E.append(P("2.2&nbsp;&nbsp;Honesty by construction", H2))
    E.append(bullets([
        "<b>Links are labels, not features.</b> Explicit references (<font name='Body-B'>#N</font>, "
        "<font name='Body-B'>owner/repo#N</font>, issue/PR URLs, commit SHAs) are scrubbed from inputs "
        "while kept as ground truth — a relation a regex can recover is a label source, never a test.",
        "<b>Frozen, cross-repo splits.</b> Train repositories are held disjoint from test "
        "repositories, so a win must generalize to unseen vocabularies and APIs; the default real-data "
        "split is temporal-by-commit-date. Splits are recorded in a dataset card.",
        "<b>Temporal leakage guard.</b> A query's optional <font name='Body-B'>as_of</font> time "
        "excludes any candidate or positive that only becomes valid later; the validator exits "
        "non-zero on leakage.",
        "<b>Hard negatives are the point.</b> Each query carries near-miss negatives (wrong file in "
        "the same package, wrong test in the same suite, plausible-but-unrelated PR); without them "
        "Recall@K flatters every model.",
        "<b>Redistribution is metadata-only.</b> Provenance points back to public URLs; full source "
        "text is not redistributed in the public repo.",
    ]))

    # ===================================================== 3. pipelines
    E.append(P("3&nbsp;&nbsp;Pipelines", H1))
    E.append(heading_rule())
    E.append(P(
        "The pipeline turns public GitHub data into a provenance-bearing, leakage-guarded, frozen "
        "benchmark, and then replays cheaply on cached features. It is methodology-as-code, and every "
        "stage is reproducible from a clean checkout with numpy alone — no GPU and no network on the "
        "evaluation path.", BODY))

    pipe = [
        ["Stage", "What it does", "Key property"],
        ["1. Ingest", "Map issue / PR / commit JSON and a PR's changed files into typed records and <i>fixes</i> / <i>modifies</i> edges (pure standard library).", "Offline by default; live fetch is opt-in behind a flag + token, polite (rate-limit-aware, paced), recorded once then replayed."],
        ["2. Scrub", "Remove explicit cross-references from input text; keep them as labels.", "Makes the task semantic, not pointer-chasing."],
        ["3. Validate", "Enforce JSON schemas, provenance completeness, referential integrity, and the temporal leakage guard over the whole data tree.", "Exits non-zero on any error; 0 errors required to proceed."],
        ["4. Embed", "Encode record text with a frozen small embedder (MiniLM-L6, 384-d) once; cache to .npz.", "Heavy step cached and (for small tiers) committed, so eval is numpy-only."],
        ["5. Graph-enrich", "Fetch fixing-PR changed files; add file/test nodes and <i>modifies</i> edges with provenance.", "Unblocks the graph track and diff→test."],
        ["6. Benchmark + eval", "Build frozen cross-repo queries with hard negatives; score Recall@K / MRR / hard-neg; apply the graph leakage guard (gold edge removed before aggregation).", "Deterministic given a seed; one change at a time vs. the control."],
        ["7. Cards", "Emit a dataset / experiment card per run; record the outcome in the §results ledger.", "Exploratory results labelled as such; cards are the source of truth."],
    ]
    E.append(table(pipe, [2.3 * cm, W - 8.1 * cm, 5.8 * cm]))
    E.append(P("Table 2. The seven-stage pipeline. CI never runs the live builder; it validates the "
               "committed snapshot and re-runs the deterministic numpy ablation. An independent audit "
               "(R13) confirmed every numpy result reproduces the committed cards byte-for-byte.", CAP))

    # ===================================================== 4. methods
    E.append(P("4&nbsp;&nbsp;Methods and Techniques", H1))
    E.append(heading_rule())
    E.append(P(
        "All systems are scored on the <i>same</i> candidate pools, so every comparison is "
        "attributable to one change. The operators form a ladder, each testing a different "
        "hypothesis about <i>where</i> the relational signal lives.", BODY))
    meth = [
        ["System / technique", "What it is", "Tests"],
        ["Vanilla cosine", "Plain cosine over bag-of-token vectors — the off-the-shelf text floor.", "The similarity baseline."],
        ["Unsupervised IDF cosine", "Cosine with corpus IDF token weights; no relation labels.", "Is a gain just a frequency effect?"],
        ["Diagonal relation metric", "Non-negative per-token weights from a margin triplet loss over train-split <i>fixes</i> pairs (reweights shared dimensions only).", "Is the relation a per-token reweighting?"],
        ["Two-tower projection", "Asymmetric low-rank query/document projections (margin triplet); can align <i>different</i> tokens across sides.", "Can a learned cross-token operator over bag-of-tokens generalize?"],
        ["Identity-init operator", "A relation map M initialized at identity (starts exactly at embedder-cosine), regularized back, on <i>frozen</i> embeddings.", "Can a safe refinement of frozen vectors help?"],
        ["<b>LoRA-tuned encoder</b>", "Low-rank adapters (r=8, α=16; ~0.48% of params) on MiniLM-L6 attention q/k/v, symmetric InfoNCE over train-repo pairs; pretrained weights untouched.", "Does the relation loss <i>inside</i> the representation generalize cross-repo?"],
        ["Typed graph lift", "Training-free typed mean-aggregation: aug(v)=normalize(α·own + (1−α)·mean over relation-neighbours); swept over α and hops.", "Does graph structure add signal beyond pairwise cosine?"],
        ["Learned R-GCN", "A 1-hop relational GCN, init at frozen features, InfoNCE.", "Does a learned GNN beat parameter-free aggregation?"],
        ["Chunking (FirstP / MaxP)", "Score by the lede chunk (FirstP) or the best chunk (MaxP) over a chunked body.", "Where is the signal — front-loaded or deep?"],
        ["SLM packer (MVP-2 seed)", "Relation-conditioned subgraph retrieval packs context for a small LM (dry-run only).", "Can a relation-packed subgraph drive an SLM?"],
    ]
    E.append(table(meth, [3.3 * cm, W - 9.6 * cm, 4.4 * cm]))
    E.append(P("Table 3. The method ladder. The bag-of-tokens and graph systems are numpy-only and "
               "deterministic; the embedder systems use a frozen MiniLM-L6 (22M params, CPU), embedded "
               "once and cached so evaluation needs no torch.", CAP))
    E.append(P("4.1&nbsp;&nbsp;Objective and metrics", H2))
    E.append(P(
        "Operators are trained with a <b>topological margin loss</b> — a triplet objective "
        "L_topo = Σ [γ − s_r(u,v⁺) + s_r(u,v⁻)]₊ — realized at scale as a symmetric InfoNCE / "
        "multiple-negatives contrastive loss with in-batch negatives. The relation score is "
        "s_r(u,v) = f_r(h_u, h_v): a diagonal W_r (per-token reweighting), a low-rank W_r = Wq·Wd "
        "(two-tower), or a translational map h_v ≈ M_r·h_u. Systems are judged by <b>Recall@K</b> "
        "(K = 1, 5, 10), <b>MRR</b>, and <b>hard-negative accuracy</b>. R@1 and MRR are the "
        "discriminating metrics; at the small candidate pools here R@5/R@10 are near-ceiling.", BODY))

    # ===================================================== 5. hypotheses
    E.append(P("5&nbsp;&nbsp;Hypotheses", H1))
    E.append(heading_rule())
    E.append(P("The program is organized around six falsifiable questions, each with a control it "
               "must beat and a current verdict from the evidence.", BODY))
    hyp = [
        ["#", "Falsifiable hypothesis", "Status"],
        ["Q1", "Fine-tuning a small embedder with the relation/contrastive loss (LoRA) beats frozen embedder-cosine cross-repo.", "<b>Confirmed</b>, and grows with scale/density."],
        ["Q2", "Link prediction over the typed SDLC graph (GNN / KG-embedding) adds signal beyond pairwise text cosine.", "<b>Partly</b>: a parameter-free lift helps issue→PR; a learned R-GCN does not beat it at pilot scale."],
        ["Q3", "Relations with weaker surface text (diff→test, log→file) show a larger relational lift than issue→PR.", "<b>Open / blocked</b> by co-change density (structure-bound at pilot)."],
        ["Q4", "Cross-repo same-bug-class retrieval benefits from relation-trained embeddings (a latent, non-hyperlinked relation).", "<b>Not yet run</b> (Track C2)."],
        ["Q5", "Scale (more repos / pairs) gives a learned head the headroom it lacked at pilot scale.", "<b>Open</b>: data scales; learned-head re-test is a GPU follow-up."],
        ["Q6", "A code-specific embedder beats a strong general one (how much does the base matter?).", "<b>Refuted as stated</b>: the axis is <i>embedding-tuned</i>, not 'code'."],
    ]
    E.append(table(hyp, [0.9 * cm, W - 6.9 * cm, 6.0 * cm]))
    E.append(P("Table 4. Open questions and current verdicts. A 'no' reroutes the program (diagnose / "
               "change base / scale); it does not stall it.", CAP))

    # ===================================================== 6. scope
    E.append(P("6&nbsp;&nbsp;Scope of Experiments", H1))
    E.append(heading_rule())
    E.append(P(
        "Experiments run as <b>waves</b> (R3–R16E), each a single change against the appropriate "
        "control, organized into parallel tracks with explicit decision gates. A gate outcome — win "
        "or honest negative — routes the next wave; scale is spent only on a method that has already "
        "shown signal.", BODY))
    tracks = [
        ["Track", "Theme", "State"],
        ["A — Representation", "Fine-tune the encoder with the relation loss (LoRA).", "<b>Won</b> (pilot → dense Tier-2); the program's core positive result."],
        ["B — Graph", "Link prediction / aggregation over the SDLC graph.", "Probe done; free-aggregation lift robust; learned GNN re-gated on scale."],
        ["C — Tasks and labels", "diff→test, log→file; cross-repo same-bug-class.", "diff→test characterized (structure-bound); same-bug-class pending."],
        ["D — Scale (on signal only)", "Multi-split CIs; more/denser repos.", "5-split confidence done; 78-repo dense Tier-2 done; 200–500 repos is GPU-scale."],
        ["E — Beyond MVP-1", "Relational SLM, GraphRAG, agent policy.", "Entry started (subgraph packer + SLM dry-run); gated behind a solid retrieval win."],
    ]
    E.append(table(tracks, [3.5 * cm, W - 11.0 * cm, 7.5 * cm]))
    E.append(P("Table 5. The five tracks and their gate state.", CAP))
    E.append(P("6.1&nbsp;&nbsp;Method invariants and the audit", H2))
    E.append(P(
        "Six invariants make results comparable: links-as-labels; frozen cross-repo splits as the "
        "headline; pretrained embeddings as the substrate with the relational contribution measured "
        "as the delta over embedder-cosine; an experiment card per run; reproducibility from a clean "
        "checkout on numpy; and honest negatives kept. An <b>independent adversarial audit (R13)</b> "
        "stress-tested the five headline claims and returned <b>sound, zero CRITICAL issues</b>: "
        "cross-repo splits are genuinely disjoint, the reference scrub leaks no gold-pair numbers, "
        "the graph leakage guard is load-bearing, and every numpy result reproduces the committed "
        "cards byte-for-byte. A CI provenance test now guards disjointness mechanically.", BODY))

    # ===================================================== 7. results
    E.append(P("7&nbsp;&nbsp;Results", H1))
    E.append(heading_rule())

    E.append(P("7.1&nbsp;&nbsp;The cross-repo substrate is settled, and the head fails", H2))
    r_sub = [
        ["System (de-referenced, cross-repo; 174 held-out queries)", "R@1", "R@5", "MRR"],
        ["IDF cosine (bag-of-tokens bar)", "0.460", "0.828", "0.624"],
        ["<b>embedder-cosine (frozen MiniLM-L6)</b>", "<b>0.592</b>", "0.920", "<b>0.728</b>"],
        ["embedder + from-scratch head", "0.190", "0.644", "0.381"],
        ["embedder + identity-init operator", "0.592", "0.931", "0.737"],
    ]
    E.append(table(r_sub, [W - 6.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm],
                   align_cols={1: TA_CENTER, 2: TA_CENTER, 3: TA_CENTER}))
    E.append(P("Table 6. Pretrained embeddings win cross-repo with no training (0.46 → 0.592). A "
               "from-scratch head on frozen vectors is actively harmful (0.190); an identity-init "
               "operator is safe but adds nothing. The relational contribution cannot be a post-hoc "
               "head on frozen features. (Earlier rungs: bag-of-tokens projections fail cross-repo — "
               "two-tower 0.241 &lt; vanilla 0.391 — because tokens do not transfer between repos; "
               "only meaning does.)", CAP))

    E.append(P("7.2&nbsp;&nbsp;The relational fine-tune wins, and the win grows with data", H2))
    r_lora = [
        ["LoRA vs. frozen, cross-repo", "Frozen R@1", "LoRA R@1", "ΔR@1", "ΔMRR", "Held-out repos"],
        ["pilot (20 repos, 182 train pairs)", "0.592", "<b>0.655</b>", "+0.07", "+0.063", "8"],
        ["scale (55 repos)", "—", "—", "+0.080", "+0.072", "14"],
        ["<b>dense Tier-2 (78 repos, ~35 q/repo)</b>", "0.515", "<b>0.629</b>", "<b>+0.114</b>", "+0.101", "32"],
        ["5-split confidence (pilot re-splits)", "—", "—", "+0.061±0.021", "+0.052±0.010", "5 splits"],
    ]
    E.append(table(r_lora, [W - 12.6 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 3.4 * cm - 0.0],
                   align_cols={1: TA_CENTER, 2: TA_CENTER, 3: TA_CENTER, 4: TA_CENTER, 5: TA_CENTER}))
    E.append(P("Table 7. A 0.48%-parameter adapter trained on CPU generalizes to unseen repositories, "
               "and a head on the tuned vectors still adds nothing (the gain is in the representation). "
               "The progression is monotone in density (pilot +0.07 → scale +0.080 → Tier-2 +0.114): "
               "more positive pairs per repo give the relation loss more to contrast. A sweep (R16D) "
               "shows +0.114 is near-tuned, not a floor — rank is saturated (r8 ≈ r16 ≈ r32, ±0.003) "
               "and harder in-batch negatives are the only remaining lever (best r16-b48: +0.126), "
               "now bounded by CPU memory.", CAP))

    E.append(P("7.3&nbsp;&nbsp;A thin graph lift stacks on top — robust, and structure-bound where it fails", H2))
    r_graph = [
        ["Graph lift (typed mean-aggregation)", "R@1", "Note"],
        ["issue→PR, frozen + graph", "0.621", "vs. frozen-cosine 0.592"],
        ["<b>issue→PR, LoRA + graph</b>", "<b>0.690</b>", "vs. LoRA-cosine 0.655; best at hops=1, α≈0.25"],
        ["learned 1-hop R-GCN (issue→PR)", "0.575", "overfits ~180 pairs; &lt; free-agg 0.609"],
        ["diff→affected-test (any α, any hops)", "0.009", "flat: 46.9% of gold tests isolated → 59.8% reachable ceiling"],
    ]
    E.append(table(r_graph, [W - 8.5 * cm, 2.0 * cm, 6.5 * cm], align_cols={1: TA_CENTER}))
    E.append(P("Table 8. The R16E α×hops sweep shows the issue→PR lift is a <b>plateau</b> (positive "
               "across α in [0, 0.75] on both frozen and LoRA features) and <b>saturates at 1 hop</b> "
               "(hops=1 ≡ hops=2, byte-for-byte) — so R11B's 0.690 is robust and cheap, not a tuned "
               "spike, and α=1.0 reproduces embedder-cosine exactly (sanity anchor). diff→test is flat "
               "at every setting because, once the gold edge is honestly removed, ~47% of positive "
               "test nodes are degree-0 — a feature- and hop-independent ceiling. The limiter is "
               "co-change <i>density</i> (a data problem), not the method.", CAP))

    E.append(P("7.4&nbsp;&nbsp;Two counter-intuitive negatives, and a chunking mirror", H2))
    E.append(bullets([
        "<b>Base model: 'embedding-tuned', not 'code'.</b> A code-MLM (codebert) collapses to R@1 "
        "0.144; the axis is monotone in embedding-tuning — codebert 0.14 &lt; unixcoder 0.45 &lt; "
        "st-codesearch 0.55 &lt; MiniLM 0.592 ≈ bge 0.598. A true code-<i>embedding</i> base "
        "(jina-code) gets the best R@5 ever (0.960) and ties MRR but does not beat top-1.",
        "<b>More text hurts.</b> A paired control de-truncating bodies (500 → 8000 chars) <i>lowers</i> "
        "every system by 0.09–0.15 R@1 (embedder 0.69 → 0.55): the first ~500 chars carry the signal "
        "and the rest dilutes it. Truncation was a feature.",
        "<b>Chunking confirms the mechanism.</b> For front-loaded issue→PR, <b>FirstP</b> (the lede) "
        "wins at every chunk size (FirstP@512 0.701 &gt; MaxP 0.668). The mirror image: for deep-signal "
        "diff→affected-test, <b>MaxP</b> wins at every chunk size (ΔR@1 up to +0.346 at small chunks), "
        "exactly as predicted.",
        "<b>Bag-of-tokens ordering is stable at scale.</b> At 55 repos IDF 0.444 ≥ vanilla 0.333, and "
        "at dense Tier-2 IDF 0.389 vs. vanilla 0.287 (+0.102) — the frequency signal generalizes.",
    ]))

    E.append(P("7.5&nbsp;&nbsp;Full results ledger", H2))
    led = [
        ["Wave", "Hypothesis (abbrev.)", "Finding", "Headline"],
        ["Synthetic per-token", "relation supervision &gt; vanilla/IDF when the link is per-token", "<b>confirmed</b>", "R@1 0.82 vs IDF 0.38 vs van. 0.11"],
        ["Cross-token synthetic", "only a cross-token operator bridges disjoint vocab", "<b>confirmed</b>", "tower R@1 0.83 vs all-else 0.01"],
        ["Real issue→PR (linked)", "does the synthetic win transfer?", "<b>no</b> — surface-rich; IDF ties diagonal", "IDF R@1 0.46"],
        ["De-ref. cross-repo (tokens)", "learned head generalizes cross-repo?", "<b>no</b> — tower 0.24 &lt; vanilla 0.39", "tower 0.24"],
        ["Embeddings cross-repo", "pretrained embeddings generalize?", "<b>yes</b> — embedder-cosine wins", "R@1 0.59 vs IDF 0.46"],
        ["Head on frozen embeddings", "does our operator add value on top?", "<b>no</b> at pilot — from-scratch 0.19; identity ties 0.59", "—"],
        ["LoRA fine-tune (A)", "relation loss inside the rep beats frozen?", "<b>YES</b> (pilot)", "R@1 0.66 vs 0.59; MRR 0.79"],
        ["Multi-split (R11A, D)", "is the LoRA win robust, not split luck?", "<b>YES</b> — positive on all 5 splits", "ΔR@1 +0.061±0.021"],
        ["Graph probe (R11B, B)", "does typed aggregation add beyond cosine?", "<b>small but real; stacks with LoRA</b>", "LoRA+graph 0.69"],
        ["Base model (R12B, Q6)", "does a code/stronger base help?", "<b>embedding-tuned matters, not 'code'</b>", "bge 0.598 / MiniLM 0.592 / codebert 0.144"],
        ["Learned R-GCN (R12B)", "does a learned GNN beat free aggregation?", "<b>no</b> at pilot — overfits ~180 pairs", "rgcn 0.575 &lt; free-agg 0.609"],
        ["Scale ~55 repos (R12A)", "does bag-of-tokens hold at scale?", "<b>yes</b> — IDF still best", "IDF 0.444 ≥ van. 0.333"],
        ["Relational SLM v0 (R12C, E)", "can a subgraph drive an SLM (MVP-2)?", "<b>dry-run runs</b> (no benchmark claim)", "retrieval ~0.9 top-5"],
        ["LoRA-at-scale (R13A, D)", "does the LoRA win hold on 55 repos?", "<b>yes — and grows</b>", "ΔR@1 +0.080 (14 test repos)"],
        ["Code base (R13B, Q6)", "does a code base beat the general substrate?", "<b>no</b> — monotone in embedding-tuning", "codebert 0.14 ... MiniLM 0.59 ≈ bge 0.60"],
        ["Full text (R14)", "does de-truncating bodies help?", "<b>no — it HURTS</b> (paired control)", "−0.09 to −0.15 R@1; embedder 0.69→0.55"],
        ["Chunked MaxP (R15)", "does MaxP beat FirstP for issue→PR?", "<b>no</b> — signal front-loaded", "FirstP@512 0.701 &gt; MaxP 0.668"],
        ["Code base pinned (R15B)", "does a true code+embedding base win?", "<b>qualified yes</b> — best R@5, ties MRR", "R@5 0.960; R@1 0.580"],
        ["Deep-content chunk (R16A)", "does MaxP win where signal is deep?", "<b>yes</b> — mirror of R15", "ΔR@1 up to +0.346"],
        ["Dense Tier-2 base (R16B, D)", "does bag-of-tokens hold at 78 dense repos?", "<b>yes</b> — density did not erase the gap", "IDF 0.389 vs van. 0.287"],
        ["LoRA-at-Tier-2 (R16C, D)", "does the LoRA win hold at dense ~80 repos?", "<b>yes — and grows further</b>", "ΔR@1 +0.114 (0.515→0.629)"],
        ["LoRA sweep (R16D, D)", "is +0.114 a floor or near-tuned?", "<b>near-tuned</b> — rank saturated", "best r16-b48 ΔR@1 +0.126"],
        ["Graph-lift sweep (R16E, B)", "tuned knife-edge or robust; can hops rescue diff→test?", "<b>robust plateau + structure-bound</b>", "issue→PR 0.690 (h1); diff→test ceiling 59.8%"],
        ["LoRA-win CIs (R17a, A/D)", "does the headline delta survive a within-split CI?", "<b>yes</b> — both query- and repo-cluster 95% CIs exclude zero; broad but not uniform (5/8 repos)", "ΔR@1 +0.063, CI [+0.006,+0.121]; ΔMRR CI [+0.027,+0.102]"],
        ["diff→test density (R17b, B)", "is the 59.8% ceiling method-bound or an ingest-depth artefact?", "<b>density artefact</b> — gold tests heavily co-changed (median 35 commits)", "reachable ceiling 59.8% → 96.4% under real co-change"],
        ["Negatives lever (R18, A/D)", "more vs harder in-batch negatives on the M5 GPU?", "<b>harder, not more</b> — random pool flat (b192/384 OOM the M5); same-repo hard batching is the new best", "repo-hard ΔR@1 +0.137, CI [+0.112,+0.164], 31/32 repos up"],
        ["Hardness multi-seed (R19, A/D)", "is the hardness gain real across seeds or MPS noise?", "<b>real</b> — paired gap positive on both seeds, tight (mean +0.020, std ~0.001)", "random +0.122 vs repo-hard +0.142; mining (H2) deferred"],
    ]
    led_w = [3.0 * cm, W - 12.2 * cm, 4.7 * cm, 4.5 * cm]
    lt = table(led, led_w, font=7.4, lead=9.0)
    # override font size on this dense table
    lt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.3), ("LEADING", (0, 0), (-1, -1), 8.9),
                            ("TOPPADDING", (0, 0), (-1, -1), 2.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4)]))
    E.append(lt)
    E.append(P("Table 9. Full results ledger (waves R3–R16E). The committed experiment cards under "
               "data/cards/examples/ are authoritative for every number.", CAP))

    # ===================================================== 8. conclusions
    E.append(P("8&nbsp;&nbsp;Conclusions", H1))
    E.append(heading_rule())
    E.append(P(
        "<b>The relational win lives in the base representation.</b> Across every experiment the "
        "single lever that consistently pays off is <i>changing the representation itself</i> — an "
        "embedding-tuned substrate, LoRA reshaping it with the relation loss, and a thin "
        "parameter-free graph lift on top. The single thing that consistently fails is <i>adding a "
        "learned head over frozen features</i> — a from-scratch tower, an identity-init operator, a "
        "learned R-GCN all overfit at pilot scale and do not help.", BODY))
    E.append(P("What is settled at pilot/Tier-2 scale:", BODY_T))
    E.append(bullets([
        "<b>Pretrained embeddings are the cross-repo substrate.</b> Meaning transfers across "
        "repositories where tokens do not; a frozen semantic embedder beats every bag-of-tokens "
        "system on held-out repos with zero training.",
        "<b>The contribution belongs in the representation</b> (LoRA), not a post-hoc head — and the "
        "win is robust (all 5 splits) and grows monotonically with per-repo density (+0.07 → +0.114).",
        "<b>The graph lift is real but thin and 1-hop</b> on issue→PR; diff→test's 59.8% "
        "ceiling is an <b>ingest-depth artefact</b> — real co-change history lifts it to 96.4% "
        "(R17b), so the blocker is removable by denser data, not the method.",
        "<b>Base = embedding-tuned, not 'code'</b>; and <b>more text is not better</b> — both "
        "counter-intuitive, both from paired controls.",
    ]))
    E.append(P(
        "What remains open is honest and bounded: the learned-head and learned-GNN re-tests at GPU "
        "scale; whether weaker-surface relations (diff→test, log→file) beat issue→PR once the data is "
        "dense enough; cross-repo same-bug-class retrieval; and the remaining paper losses "
        "(L_contrastive, L_rel, L_graph, L_logic, L_align). The lab's enduring deliverable is the "
        "frozen public benchmark with honest baselines and full provenance that makes exactly these "
        "questions measurable.", BODY))

    # ===================================================== 9. next steps
    E.append(P("9&nbsp;&nbsp;Next Steps", H1))
    E.append(heading_rule())
    nxt = [
        ["Direction", "Concrete next action", "Gate / prerequisite"],
        ["Scale the LoRA win", "Tier-2 full breadth (200–500 repos); harder/larger in-batch negative pools (the one lever R16D left).", "GPU memory (CPU-bound today)."],
        ["Learned graph model", "Richer/regularized multi-hop R-GCN on pretrained node features; KG-embedding scoring (DistMult/ComplEx/RotatE).", "Track-D scale for supervision; must beat ~0.690 (LoRA) / ~0.621 (frozen), not raw cosine."],
        ["Unblock diff→test", "R17b showed the 59.8% ceiling is an ingest artefact (real ceiling 96.4%); next is the retrieval re-eval on a denser graph with embedded PR nodes.", "Torch to embed new nodes — the ceiling no longer blocks it."],
        ["New tasks/labels", "Activate log→file; cross-repo same-bug-class via a curated SWE bug-class ontology (labels independent of body text).", "Non-degenerate (not regex-recoverable) before it counts."],
        ["Long-text handling", "Salient-section selection / better pooling — not naive whole-body (which hurts).", "Beat the FirstP lede baseline."],
        ["Relational SLM (MVP-2+)", "QLoRA SLM with relation/policy heads (review, test selection, risk) + GraphRAG packer; outcome learning.", "Gated behind a solid retrieval win — which Track A now provides."],
    ]
    E.append(table(nxt, [3.4 * cm, W - 11.0 * cm, 7.6 * cm]))
    E.append(P("Table 10. The decision-gated next steps. Per the program's compute discipline, GPU and "
               "broad scale are spent only where a method has already shown signal.", CAP))
    E.append(Spacer(1, 8))
    E.append(HRFlowable(width="100%", thickness=0.7, color=RULE, spaceAfter=5))
    E.append(P(
        "<i>All results are exploratory and pilot/Tier-2 scale; they are signals to confirm, not "
        "release-quality evidence. Where this report and the repository's committed experiment cards "
        "disagree, the cards are authoritative.</i>", SMALL))

    doc.build(E)
    return OUT


if __name__ == "__main__":
    path = build()
    print("WROTE", path, os.path.getsize(path), "bytes")
