# `v1/knowledge_graph/`: disease knowledge base and retrievers of the v1 application

Part of v1, the original system (2025 to April 2026). One JSON knowledge base of the seven conditions and two Python
retrievers that the Gradio application [`../demo_v5.py`](../demo_v5.py) uses for its Knowledge Graph tab, for
follow-up questions and for the consistency check of its reports. Revival-era change: on 31 October 2026 the
consistency check in `rag_retriever.py` became negation-aware and its list of hallmark phrases went from 18 to 16
(the same commit changed the retry path in `demo_v5.py`).

## Files

| File | What it is |
|---|---|
| [`disease_knowledge_base.json`](disease_knowledge_base.json) | the knowledge base, version 3.0 (metadata: last updated 24 March 2026), stored with Git LFS: seven entries (`drought`, `overwatering`, `root_rot`, `frost_injury`, `gray_mold`, `white_mold`, `healthy`), each with a display name, severity, symptoms, causes, treatments, recovery timeline, prevention, visual indicators, part-specific treatments keyed by the detector's six parts, and sources (3 or 4 per entry, 24 in all, with URLs); general care practices (watering, mulching, spacing, monitoring, sanitation, nutrition); metadata listing seven primary sources (UC IPM and six university extension services) |
| [`rag_retriever.py`](rag_retriever.py) | `DiseaseRAG`: look-ups by condition (classifier labels such as `overwatered` or `frost` are mapped to the keys), graph data for the Knowledge Graph tab, and `check_definite_hallucination()`, the consistency check |
| [`qa_retriever.py`](qa_retriever.py) | `DiseaseQA`: answers a follow-up question from one section of the knowledge base, chosen by keyword patterns (recovery, causes, treatments, prevention, visual signs, severity, or a comparison of two conditions), and suggests follow-up questions |
| [`__init__.py`](__init__.py) | exports `DiseaseRAG` and `DiseaseQA` |

The entries were compiled by hand from the cited university extension publications, with a language model helping
with the wording and without automated collection ([MANUSCRIPT_QA.md](../../docs/MANUSCRIPT_QA.md), item 2.11). The
JSON and both retrievers were committed on 21 January 2026, in the commit that deleted the FAOSTAT files
(`kg_nodes_faostat.csv`, `kg_relationships_faostat.csv`) described by an earlier version of this page; version 3.0 of
the knowledge base was committed on 12 April 2026.

## How the application uses it

- **Knowledge Graph tab:** `get_graph_data()` (all seven conditions with up to five symptoms, three treatments and
  three causes each) and `get_disease_graph_data()` (one condition with all of its symptoms, treatments, causes and
  recovery timeline).
- **Consistency check:** after the first report for a photograph, `check_definite_hallucination()` looks for hallmark
  phrases of the other conditions, sentence by sentence. A phrase does not count when a negation cue (no, not,
  without, absence of, free of, lack of, unlike, rather than, instead of, and others) stands in the five words before
  it or a phrase such as "is not" or "are absent" follows it; such mentions are returned as `negated_mentions`. A
  flagged report is regenerated once with a stricter prompt; if it is flagged again, the application appends a
  warning. The 16 phrases:

  | Condition | Hallmark phrases |
  |---|---|
  | drought | cracked soil, limp petioles |
  | overwatering | mushy crown, algae growth, waterlogged |
  | root rot | foul odor, red stele |
  | frost injury | black eye, translucent leaves, glassy tissue |
  | gray mold | gray fuzz, dusty spores, tan mummies |
  | white mold | white cotton, black sclerotia, cottony mycelium |

  "brown roots" and "botrytis" were removed on 31 October 2026 because they flagged correct reports; healthy has no
  phrases ([MANUSCRIPT_QA.md](../../docs/MANUSCRIPT_QA.md), item 2.7).
- **Follow-up questions:** `DiseaseQA.answer_question()` answers a question about the last diagnosis directly when its
  pattern score gives a confidence of at least 0.3, and adds the answer to the prompt as "Relevant knowledge" when a
  question comes with a photograph (confidence at least 0.5). `suggest_questions()` adds up to three suggestions after
  a diagnosis other than healthy.

## Running

Both retrievers need only the Python standard library and read the JSON from their own folder (pass `kb_path=` to use
another file). Each file has a small self-test:

```bash
cd v1
python knowledge_graph/rag_retriever.py    # look-ups, graph sizes, a sample check
python knowledge_graph/qa_retriever.py     # sample questions and suggestions
```

## Dependencies

- **Used by v1:** [`../demo_v5.py`](../demo_v5.py); [`../hallucination_evaluation/`](../hallucination_evaluation/)
  reads the JSON (expected symptoms for its grounding score); the recovered April 2026 scoring script
  [`score_minigpt_rubric.py`](../recovered/labbox_2026-10-19/paper/scripts/score_minigpt_rubric.py) reads the
  JSON.
- **Used by v2:** [`../../v2/vlm/latency_v1.py`](../../v2/vlm/latency_v1.py) and
  [`../../v2/vlm/minigpt_reports.py`](../../v2/vlm/minigpt_reports.py) import `DiseaseRAG` to run the consistency
  check and the regeneration as the application does; they import it from the working folder that
  [`../../v2/infra/make_workdir.sh`](../../v2/infra/make_workdir.sh) builds.
- **Used by the 2026 manuscript's figure scripts:**
  [`../../paper/figure_scripts/fig_knowledge_base.py`](../../paper/figure_scripts/fig_knowledge_base.py) and
  [`../../paper/figure_scripts/fig_symptom_overlap.py`](../../paper/figure_scripts/fig_symptom_overlap.py) read the
  JSON from an absolute local path (`D:/agai-paper/worktree/knowledge_graph/...`).

## Known problems

- **Unused parts of the knowledge base.** No code in the repository calls `get_part_specific_treatments()`,
  `get_context()`, `get_detailed_context()`, `get_sources()`, `get_best_practices()` or `check_hallucination()`, so
  the application shows neither the part-specific treatments nor the sources, and injects no knowledge-base context
  into the first report's prompt.
- **Git LFS.** Without `git lfs pull` the JSON is a pointer file. `DiseaseRAG` then prints a parse error and both
  retrievers continue with an empty knowledge base: the application starts, but the graph is empty and the check
  never flags.
- **Labels.** The keys differ from the classifier's folder names (`overwatering`, `frost_injury` against
  `overwatered`, `frost`). `DiseaseRAG` maps both; `DiseaseQA` does not map `overwatered` (the application passes
  its display labels, such as `overwatering` and `frost injury`, which it does map).
