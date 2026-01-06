# SUJBOT2 - TODO List

## TIER 1: SOTA RAG Improvements (High Priority)

### Corrective & Self-Reflective RAG
- [ ] **Corrective RAG (CRAG)** - Retrieval evaluator + corrective loop
  - Confidence scoring po retrieval (correct/incorrect/ambiguous)
  - Pro incorrect: trigger alternate retrieval nebo web search
  - Reference: [CRAG Paper (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [ ] **Self-Reflection Loop** - Agent self-critique pred final answer
  - Reflection step v agent_base.py
  - Max 2 reflection iterations
  - Reference: [Self-RAG](https://blog.langchain.com/agentic-rag-with-langgraph/)

### Retrieval Pipeline Upgrades
- [ ] **Late Chunking** - Jina-style contextual chunk embeddings (+24% recall)
  - Zpracovat cely dokument transformerem pred chunking
  - Model: jina-embeddings-v3 nebo jina-colbert-v2
  - Reference: [Jina Late Chunking (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [ ] **Enable Cross-Encoder Reranking** - jina-reranker-v3 (61.85 NDCG@10)
  - Currently disabled v config - enable
  - Reference: [jina-reranker-v3 (2025)](https://arxiv.org/abs/2509.25085)
- [ ] **Query Router** - Semantic routing na optimalni pipeline
  - Klasifikovat query typ (simple/multi-hop/compliance/synthesis)
  - Route na optimalni retrieval strategy
  - Reference: [RAGRouter Paper (2025)](https://arxiv.org/abs/2505.23052)

### Evaluace
- [ ] **Validovat evaluation dataset** 50+ prikladu
  - Nuclear reactor specs: 10
  - Legal definitions: 10
  - Safety regulations: 10
  - Cross-document synthesis: 10
  - Edge cases (negation, multi-hop): 10

---

## TIER 2: Dulezite (Medium Priority)

### Retrieval Improvements
- [ ] Multi-Query RAG Fusion (beyond HyDE)
- [ ] Aktivovat graph-enhanced retrieval v standard workflows
- [ ] MMR deduplication v retrieval (diversity ranking)
- [ ] Experiment: optimal graph_boost_weight (0.2 vs 0.3 vs 0.4)
- [ ] DRM - document retrieval mismatch - nastudovat a pridat

### Agents & Tools
- [ ] HITL implementation (clarification requests, approval flows)
- [ ] Zkontrolovat tools
- [ ] Todo list middleware pro agents
- [ ] Similar chunks tool
- [ ] Co dela extractor agent kdyz mu pretece kontext?

### Dokumenty a Extrakce
- [ ] Zpracovavat obrazky + tabulky
- [ ] Naladit extrakci dokumentu - Vendy + Petr Vecer

### Knowledge Graph
- [ ] Naladit a vylepsit knowledge graph
- [ ] Vazby do knowledge graphu - navrhnout nebo pozadat SUJB
- [ ] Predstavit lidem ze SUJB knowledge graph, vytahnout temata (expertyzy)

---

## TIER 3: Nice-to-Have (Lower Priority)

### Advanced Features
- [ ] Streaming partial results
- [ ] Agent performance ranking
- [ ] Adaptive chunk size per document type
- [ ] ColBERT late interaction retrieval
- [ ] Cost-aware tool selection
- [ ] Mene agentu je vice - zmensit z 8 na mene

### Admin a UI
- [ ] Moznost pridani dokumentu administratorem

### Uzivatelske Funkce
- [ ] Kazdy uzivatel at ma custom memories
- [ ] Vymyslet co dat do State - seznam dokumentu (id + topic)

### SUJB Spoluprace
- [ ] Pobavit se s jednotlivymi uredniky a zjistit jejich workflows - brezen
- [ ] Ziskat seznam tematickych celku (odpovednosti)

### Infrastruktura
- [ ] Sjednotit github na separatni ucet a vycistit od datasetu

---

## Completed
- [x] Doimplementovat podporu pro vsechny formaty (pdf, docx, txt, md, latex)
- [x] Empty JSON kdyz tool nic nenajde - ma vyznam
- [x] LLM as a judge - first rationale then score (binary) - structured output
- [x] Moznost rozklikavat konverzace uzivatelu na admin page
- [x] Moznost u kazde zpravy agenta dat palec nahoru/dolu
- [x] Section preview pri cursor hover

---

## Research Zdroje

### CRAG & Self-RAG
- [Corrective RAG (CRAG)](https://www.kore.ai/blog/corrective-rag-crag)
- [Self-Reflective RAG with LangGraph](https://blog.langchain.com/agentic-rag-with-langgraph/)
- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)

### Late Chunking & Embeddings
- [Jina Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [jina-embeddings-v3](https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/)

### Query Routing
- [RAGRouter Paper](https://arxiv.org/abs/2505.23052)
- [Semantic Router for RAG](https://medium.com/@talon8080/mastering-rag-chabots-semantic-router-user-intents-ef3dea01afbc)

### Reranking
- [jina-reranker-v3 (SOTA 2025)](https://arxiv.org/abs/2509.25085)
- [ColBERT v2](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)

### Legal RAG
- [LegalBench-RAG Benchmark](https://arxiv.org/abs/2408.10343)
- [LRAGE Evaluation Tool](https://arxiv.org/html/2504.01840v1)

### RAG Surveys
- [Comprehensive RAG Survey 2024](https://arxiv.org/abs/2410.12837)
- [Reasoning Agentic RAG Survey](https://arxiv.org/html/2506.10408v1)
