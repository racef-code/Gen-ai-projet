# Système d'Analyse Intelligente de Documents — Documentation Technique

> **Ce document est écrit pour le développeur du projet.** Il explique l'architecture interne, les mathématiques sous-jacentes et les décisions de conception. Pas de marketing, que de la technique.

---

## Table des matières

1. [Le flux de données (Data Pipeline)](#1-le-flux-de-données-data-pipeline)
2. [Les mathématiques des analyses](#2-les-mathématiques-des-analyses)
3. [Le casse-tête incrémental](#3-le-casse-tête-incrémental)
4. [Architecture des dossiers](#4-architecture-des-dossiers)
5. [Roadmap d'implémentation](#5-roadmap-dimplémentation)

---

## 1. Le flux de données (Data Pipeline)

### Vue d'ensemble

```
[Fichier (PDF/TXT/DOCX/MD) ou URL]
         │
         ▼
    EXTRACTION        ingestion/loaders.py | ingestion/scraper.py
         │  texte brut
         ▼
    NETTOYAGE         _clean_text() / _clean_scraped_text()
         │  texte propre
         ▼
    CHUNKING          ingestion/chunker.py
         │  liste de Chunk(text, doc_id, chunk_index, metadata)
         ▼
    EMBEDDING         ingestion/embedder.py → Ollama (nomic-embed-text)
         │  liste de vecteurs float[768]
         ▼
    STOCKAGE VECTO.   persistence/vector_store.py → ChromaDB (sur disque)
    STOCKAGE MÉTA.    persistence/metadata_store.py → SQLite (sur disque)
         │
         ▼
    TOPIC UPDATE      persistence/incremental.py → BERTopic.partial_fit()
```

---

### Étape 1 — Extraction du texte brut

**Fichiers :** `ingestion/loaders.py` et `ingestion/scraper.py`

**Fichiers locaux (`loaders.py`) :**

- **PDF** : `pypdf.PdfReader` lit les bytes page par page. Il extrait le texte via le flux PDF interne (pas d'OCR). Un PDF scanné retournera une chaîne vide — cas géré avec un warning.
- **DOCX** : `python-docx` décompresse l'archive ZIP interne du `.docx` et parse les balises XML `<w:p>` (paragraphes). Il itère sur `doc.paragraphs` et concatène avec `\n\n`.
- **TXT / Markdown** : décodage UTF-8 avec fallback latin-1. Le Markdown est traité comme du texte brut — les balises `**`, `#`, `[]()` restent dans la chaîne (choix intentionnel : pas de perte sémantique liée au rendu).

**Scraping web (`scraper.py`) :**

1. `requests.get()` télécharge le HTML brut avec un User-Agent neutre.
2. `BeautifulSoup` parse l'arbre DOM avec le parser `lxml` (plus rapide que `html.parser`).
3. Les balises de bruit (`<nav>`, `<footer>`, `<aside>`, `<script>`, `<style>`, `<noscript>`, etc.) sont **décomposées** (`tag.decompose()`) — elles sont retirées de l'arbre avant l'extraction.
4. Sélection du contenu principal dans l'ordre : `<main>` → `<article>` → `[role='main']` → `.content` → `#content` → `<body>` (fallback).
5. `.get_text(separator="\n")` extrait le texte en conservant les sauts de ligne structurels.

**Nettoyage commun :**
- `CRLF → LF`
- Réduction des séquences de lignes vides (> 2 consécutives → 2 max)
- `text.strip()`

---

### Étape 2 — Chunking (découpage intelligent)

**Fichier :** `ingestion/chunker.py`

Un modèle d'embedding a une fenêtre de contexte maximale. `nomic-embed-text` accepte jusqu'à ~8192 tokens, mais un chunk trop long dilue la sémantique. La bonne pratique est de rester entre 128 et 512 tokens par chunk.

**Stratégie 1 : `paragraph`** (défaut recommandé)

```python
paragraphs = re.split(r"\n{2,}", text)
```

Un double `\n` est une frontière sémantique naturelle. Les paragraphes trop courts (`< MIN_CHUNK_CHARS = 80 chars`) sont fusionnés avec le suivant dans un buffer cumulatif pour éviter les micro-chunks qui noieraient la recherche vectorielle.

**Stratégie 2 : `sliding_window`**

```
Texte :  [w1 w2 w3 w4 w5 w6 w7 w8 w9 w10]   CHUNK_SIZE=5, OVERLAP=2

Chunk 0: [w1 w2 w3 w4 w5]
Chunk 1:          [w4 w5 w6 w7 w8]           ← 2 mots de chevauchement
Chunk 2:                   [w7 w8 w9 w10]
```

Le **chevauchement** (overlap) garantit qu'une idée à cheval sur deux chunks n'est pas coupée sans contexte. En RAG, quand la question porte sur `w5`, le chunk 0 ET le chunk 1 sont candidats — le retriever a deux chances de trouver le bon passage.

> **Note :** on utilise les mots (`str.split()`) comme approximation des tokens. Ratio empirique : 1 token ≈ 0.75 mot en français/anglais.

**Objet `Chunk` produit :**

```python
@dataclass
class Chunk:
    text: str          # texte brut du fragment
    doc_id: str        # UUID du document parent (généré à l'ingestion)
    chunk_index: int   # position dans le doc (0-based, séquentiel)
    metadata: dict     # {strategy, source, source_type, chunk_index, ...}
```

---

### Étape 3 — Embedding (vectorisation locale)

**Fichier :** `ingestion/embedder.py`

**Modèle :** `nomic-embed-text` via Ollama → vecteurs de dimension **768**.

Sous le capot, `OllamaEmbeddings` de LangChain appelle l'API REST locale :

```
POST http://localhost:11434/api/embeddings
Body: {"model": "nomic-embed-text", "prompt": "texte du chunk"}
Réponse: {"embedding": [0.023, -0.14, 0.87, ...]}  ← 768 floats
```

Ce vecteur encode **la sémantique** du texte. Intuitivement : des textes qui parlent de la même chose ont des vecteurs qui pointent dans la même direction dans l'espace à 768 dimensions.

**Batching** : les chunks sont envoyés par paquets de 32 (`_EMBED_BATCH_SIZE`) pour ne pas saturer Ollama avec des requêtes simultanées.

**Règle d'or :** les embeddings sont calculés **une seule fois**, stockés dans ChromaDB, et **jamais recalculés** — même quand on met à jour le topic model. C'est le fondement du système incrémental.

---

### Étape 4 — Stockage vectoriel (ChromaDB)

**Fichier :** `persistence/vector_store.py`

ChromaDB stocke 4 éléments par chunk :

| Champ | Exemple | Rôle |
|-------|---------|------|
| `id` | `"a3f2...8b__0"` | Clé primaire = `doc_id + "__" + chunk_index` |
| `embedding` | `[0.023, -0.14, ...]` | Vecteur float[768] |
| `document` | `"Les transformers..."` | Texte brut (pour le contexte RAG) |
| `metadata` | `{doc_id, source, ...}` | Métadonnées filtrables |

**Structure physique sur disque :**
```
data/chroma_db/
├── chroma.sqlite3          ← index des collections, métadonnées
└── <uuid>/
    ├── data_level0.bin     ← couche 0 de l'index HNSW
    └── header.bin
```

**L'index HNSW (Hierarchical Navigable Small World) :**

Pour retrouver les 5 chunks les plus proches d'une question, il faudrait calculer la distance cosinus avec **chaque** chunk stocké (coût O(N × 768) multiplications). Avec 10 000 chunks, c'est 7,68 millions d'opérations — trop lent.

HNSW construit un **graphe multi-couches** : les couches supérieures contiennent peu de nœuds (pour naviguer vite en gros) et la couche 0 contient tous les nœuds (pour affiner). La recherche est en **O(log N)** en moyenne.

```
Couche 2 :  A ————— F                   (peu de nœuds, grandes liaisons)
Couche 1 :  A — C — F — H              (densité moyenne)
Couche 0 :  A-B-C-D-E-F-G-H-I-J-K     (tous les chunks)

Requête : partir de A en couche 2, descendre vers F, raffiner en couche 0
```

**Pourquoi `upsert` et non `insert` ?**

`upsert` = update si l'ID existe, insert sinon. Si l'utilisateur réingère le même fichier modifié, les anciens chunks sont remplacés sans laisser de doublons fantômes.

---

### Étape 5 — Le pipeline RAG (Retrieval-Augmented Generation)

**Fichiers :** `agents/rag_agent.py`, `agents/query_handler.py`

```
Question utilisateur : "Quels sont les avantages des transformers ?"
         │
         ▼
embed_query(question)          ← même modèle nomic-embed-text
    → vecteur_question [768 floats]
         │
         ▼
ChromaDB HNSW search           ← top_k=5 voisins par similarité cosinus
    → chunks[0..4] : [{text, metadata, distance}, ...]
         │
         ▼
Construction du contexte
    "[Extrait 1 — Source: rapport.pdf]
     Les transformers ont révolutionné...
     ---
     [Extrait 2 — Source: article.txt]
     BERT et GPT sont des architectures..."
         │
         ▼
ChatOllama.invoke(prompt_rag)  ← LLM local (llama3 / mistral)
         │
         ▼
Réponse + sources (pour le panneau de transparence)
```

**Pourquoi embedder la question ?**

La recherche dans ChromaDB est **vectorielle**, pas textuelle. Elle ne cherche pas les mots exacts de la question dans les chunks. Elle cherche les chunks dont le *sens* est le plus proche du *sens* de la question. Si la question est "avantages des transformers" et qu'un chunk parle de "bénéfices de l'architecture attention", la similarité cosinus sera élevée même sans mot commun.

**Le prompt RAG (instruit, pas créatif) :**

```
System: "Tu réponds UNIQUEMENT d'après les extraits fournis dans le contexte.
         Si l'info n'y est pas, dis-le clairement. Ne fabrique rien."
         CONTEXTE : [5 chunks récupérés par ChromaDB]
Human:   "Quels sont les avantages des transformers ?"
```

La `temperature=0.1` du LLM est intentionnellement basse : on veut des réponses factuelles et reproductibles, pas créatives.

---

## 2. Les mathématiques des analyses

### 2.1 La similarité cosinus

**Fichier :** `analytics/similarity.py`

La similarité cosinus mesure l'angle entre deux vecteurs — pas leur distance euclidienne. C'est crucial pour les embeddings, car deux textes peuvent avoir des vecteurs de norme différente (longueurs différentes) mais pointer dans la même direction sémantique.

**Formule :**

```
              A · B           Σ(Aᵢ × Bᵢ)
sim(A, B) = ———————— = ——————————————————————
            ‖A‖ × ‖B‖   √(ΣAᵢ²) × √(ΣBᵢ²)

Résultat : [-1, +1]   (en pratique [0, 1] pour les embeddings)
```

- `1.0` → vecteurs identiques (même document ou paraphrase exacte)
- `0.0` → vecteurs orthogonaux (sujets complètement différents)
- Valeurs typiques : 0.7-0.9 = très similaires, 0.3-0.5 = liés, < 0.2 = indépendants

**Implémentation dans le code :**

```python
# 1. Normalisation L2 de chaque vecteur (divise par sa norme)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # shape: (N, 1)
normalized = matrix / norms                             # chaque ligne = vecteur unitaire

# 2. Produit scalaire matriciel = cosinus (car vecteurs unitaires)
sim = normalized @ normalized.T                        # shape: (N, N)

# 3. Clamp dans [0, 1] (les embeddings sont quasi-positifs)
sim = np.clip(sim, 0.0, 1.0)
```

**Pour la heatmap :** on calcule le **centroïde** (vecteur moyen) de tous les chunks d'un document, puis la similarité entre centroïdes. Cela donne une vision document-niveau, pas chunk-niveau.

```python
centroide_doc_A = mean([vecteur_chunk_0, vecteur_chunk_1, ..., vecteur_chunk_N])
```

---

### 2.2 Les fréquences de mots (Bar Chart + Word Cloud)

**Fichier :** `analytics/text_stats.py`

Pipeline de comptage :

```
texte brut
    │ re.sub(r"[^\w\s]", " ")  ← supprimer la ponctuation
    │ .lower().split()          ← tokenisation naïve par whitespace
    │ filtre isalpha()          ← exclure chiffres purs et underscores
    │ len(w) >= 3               ← exclure les mots trop courts
    │ w not in STOPWORDS        ← exclure articles, prépositions...
    ▼
Counter({mot: fréquence}).most_common(top_n)
```

Les **stopwords** (199 mots FR + EN) sont embarqués directement dans le code — zéro dépendance NLTK, zéro download. Cela inclut les articles (`le`, `the`), prépositions (`dans`, `with`), pronoms, auxiliaires et conjonctions.

**Word Cloud :** la bibliothèque `wordcloud` génère une image PIL en positionnant les mots proportionnellement à leur fréquence, en évitant les collisions. La `colormap="viridis"` donne un dégradé vert-jaune.

---

### 2.3 Topic Modeling : de la phrase à la carte 2D

**Fichier :** `analytics/topic_model.py`

C'est la partie la plus complexe du système. Voici la chaîne complète :

#### Étape A — Les embeddings (déjà calculés)

On récupère directement depuis ChromaDB les vecteurs float[768] de tous les chunks. Aucun recalcul.

```python
raw = get_all_embeddings()  # → {ids, embeddings, documents, metadatas}
embeddings = np.array(raw["embeddings"])  # shape: (N_chunks, 768)
```

#### Étape B — UMAP haute dimension (pour le clustering)

768 dimensions, c'est trop pour HDBSCAN (malédiction de la dimensionnalité). UMAP réduit à 10 dimensions en préservant la structure locale des voisinages :

```python
umap_hd = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine")
embeddings_10d = umap_hd.fit_transform(embeddings)  # shape: (N, 10)
```

- `n_neighbors=15` : chaque point considère ses 15 voisins les plus proches pour construire le graphe de similarité local.
- `min_dist=0.0` : les clusters sont compacts (pas d'espacement artificiel).
- `metric="cosine"` : cohérent avec ChromaDB.

#### Étape C — HDBSCAN (clustering dans les 10 dimensions)

HDBSCAN (Hierarchical Density-Based Spatial Clustering) détecte automatiquement les clusters sans qu'on spécifie leur nombre :

```
Points denses ──→ cluster    Points isolés ──→ topic -1 (outliers)
```

Le paramètre `min_topic_size=3` fixe la taille minimale d'un cluster. HDBSCAN est piloté **par BERTopic** en interne — on ne l'appelle pas directement.

#### Étape D — UMAP 2D (pour la visualisation seulement)

Un second UMAP, cette fois vers 2 dimensions, pour le scatter plot :

```python
umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine")
coords_2d = umap_2d.fit_transform(embeddings)  # shape: (N, 2)
# → chaque chunk = un point (x, y) dans le plan
```

- `min_dist=0.1` (vs 0.0 pour le clustering) : léger espacement pour que la carte soit lisible.

#### Résultat final

Chaque chunk a :
- Un topic ID (entier, `-1` = outlier)
- Des coordonnées 2D `(x, y)` pour le scatter plot
- Un label descriptif (généré par le LLM)

```
chunk_i → topic=2 → label="Machine Learning" → point (3.2, -1.7) sur la carte
```

---

#### Pourquoi BERTopic plutôt que LDA ?

| Critère | LDA (classique) | BERTopic |
|---------|-----------------|----------|
| Représentation sémantique | Bag-of-words (TF-IDF) | Embeddings neuronaux |
| Sensibilité à la langue | Faible | Forte (multilingue) |
| Nombre de topics | Fixé à l'avance | Automatique |
| Mise à jour incrémentale | Non | Oui (`partial_fit`) |
| Qualité sur petits corpus | Mauvaise | Bonne |

LDA suppose que chaque document est un mélange de topics et que chaque topic est une distribution sur les mots. BERTopic ne fait pas ces hypothèses — il groupe les documents par proximité dans l'espace des embeddings, puis caractérise chaque cluster par ses mots représentatifs (c-TF-IDF).

---

## 3. Le casse-tête incrémental

### Problème de fond

Un topic model classique se réentraîne sur **tout le corpus** à chaque modification. Si on a 1 000 documents ingérés et qu'on en ajoute un 1 001ème :

- **Approche naïve :** re-embedder les 1 001 docs + re-fit UMAP + re-fit HDBSCAN → ~15 min
- **Notre approche :** 0 re-embedding + partial_fit sur 1 doc + UMAP.transform → ~5s

### La solution en 3 couches

#### Couche 1 — Embeddings : zéro recalcul (ChromaDB)

```
Nouveau doc ajouté
       │ embed_chunks() ← SEULEMENT les nouveaux chunks
       ▼
  ChromaDB.upsert()    ← stockage permanent

get_all_embeddings()   ← récupère TOUT (anciens + nouveaux vecteurs)
                          sans aucun appel à Ollama
```

ChromaDB est la "banque de vecteurs". L'opération coûteuse (appel à Ollama) n'est faite qu'une seule fois par chunk, à l'ingestion.

#### Couche 2 — Topic Model : partial_fit (BERTopic)

```python
# Identifier les nouveaux chunks
new_indices = [i for i, did in enumerate(chunk_doc_ids) if did in new_doc_ids]
new_embeddings = embeddings[new_indices]
new_docs = [documents[i] for i in new_indices]

# Mise à jour partielle du modèle
state.topic_model.partial_fit(new_docs, new_embeddings)
```

`partial_fit` utilise sous le capot un algorithme de clustering **online** (River/MiniBatchKMeans) qui ajuste les centroïdes de clusters existants et en crée de nouveaux si nécessaire, sans revoir les données anciennes.

**Limite connue :** la qualité des clusters se dégrade légèrement avec des `partial_fit` successifs (le modèle ne "voit" jamais l'ensemble du corpus d'un coup). Si la qualité est insatisfaisante, le bouton "Forcer le re-calcul" dans l'UI déclenche un `full_fit` sur tous les embeddings stockés.

#### Couche 3 — Projection 2D : transform, pas fit_transform

```python
# Anciens chunks : déjà projetés, on les RE-PROJETTE tous avec le même modèle UMAP
coords_2d = state.umap_model.transform(embeddings_all)  # ← transform, pas fit_transform
```

`umap.transform()` projette de nouveaux points dans un espace UMAP **déjà calculé**, sans modifier le modèle. C'est rapide (pas de réoptimisation du graphe). Les nouvelles coordonnées sont cohérentes avec les anciennes.

Si `transform` échoue (trop de nouveaux points par rapport à la taille d'entraînement), on bascule vers un `fit_transform` complet sur tous les vecteurs — ce qui reste rapide car les vecteurs sont déjà en mémoire.

### Cycle de vie complet d'un document

```
Ingestion doc N+1
    │
    ├─ 1. compute_text_hash(text)          → hash SHA-256
    │      └─ is_duplicate(hash)?  → OUI : stop, avertissement UI
    │                               → NON : continuer
    │
    ├─ 2. chunk_text()                     → nouveaux Chunk[]
    ├─ 3. embed_chunks()                   → nouveaux vecteurs (Ollama, 1 seule fois)
    ├─ 4. vector_store.add_chunks()        → stockage ChromaDB permanent
    ├─ 5. metadata_store.save_document()   → SQLite (nom, hash, stats)
    │
    └─ 6. run_incremental_update()
           │
           ├─ get_all_embeddings()         → récupère TOUT depuis ChromaDB
           ├─ BERTopic.partial_fit()       → met à jour les clusters
           ├─ UMAP.transform()             → re-projette en 2D
           ├─ Nouveaux topics détectés ?
           │     └─ OUI : auto_label_topics() → appel LLM → label court
           └─ Sauvegarder state → pickle sur disque
```

### Persistance inter-sessions

Au redémarrage de l'app (`streamlit run app/main.py`), `init_session_state()` restaure :

| Source | Ce qui est restauré |
|--------|---------------------|
| `data/metadata.db` (SQLite) | Liste des documents ingérés (noms, stats, hashes) |
| `data/chroma_db/` (ChromaDB) | Tous les vecteurs + textes des chunks |
| `data/topic_model/topic_state.pkl` (pickle) | Modèle BERTopic, coordonnées UMAP 2D, labels LLM |

L'application est donc **entièrement stateless** d'une session à l'autre : relancer `streamlit run` redonne exactement le même état qu'à la fermeture.

---

## 4. Architecture des dossiers

```
Gen-ai-projet/
├── app/                    ← Noyau de l'application
│   ├── main.py             ← Point d'entrée unique (st.set_page_config + routing)
│   ├── config.py           ← Toutes les constantes (chemins, modèles, paramètres)
│   └── state.py            ← Initialisation + restauration du session_state
│
├── agents/                 ← Intelligence (LLM + RAG)
│   ├── rag_agent.py        ← Chain LangChain : embed → retrieve → LLM → réponse
│   └── query_handler.py    ← Orchestration : validation, erreurs, persistance historique
│
├── ui/                     ← Interfaces Streamlit (uniquement du rendu)
│   ├── ingestion_ui.py     ← Page ingestion : upload, URL, preview, pipeline
│   ├── chat_ui.py          ← Page chat : bulles, panneau transparence, filtres
│   └── analytics_ui.py    ← Page dashboard : 5 onglets, cache, refresh
│
├── ingestion/              ← Pipeline de transformation texte → vecteurs
│   ├── loaders.py          ← PDF/TXT/DOCX/MD → texte brut
│   ├── scraper.py          ← URL → texte propre (BeautifulSoup)
│   ├── chunker.py          ← texte → liste de Chunk (paragraph / sliding_window)
│   └── embedder.py         ← Chunk[] → float[768][] via Ollama
│
├── analytics/              ← Analyses et visualisations
│   ├── text_stats.py       ← Fréquences de mots + stopwords
│   ├── topic_model.py      ← BERTopic incrémental + UMAP + auto-label LLM
│   ├── similarity.py       ← Matrice cosinus inter-documents
│   └── visualizations.py  ← 5 figures Plotly + image PIL (word cloud)
│
├── persistence/            ← Accès aux données persistantes
│   ├── vector_store.py     ← Interface ChromaDB (add, query, delete, get_all)
│   ├── metadata_store.py   ← Interface SQLite (documents, chat_history)
│   └── incremental.py      ← Orchestrateur post-ingestion (hash, dedup, update)
│
└── data/                   ← Données locales (ignorées par git)
    ├── chroma_db/          ← Index HNSW + SQLite interne de ChromaDB
    ├── topic_model/        ← topic_state.pkl (BERTopic + UMAP sérialisés)
    └── metadata.db         ← SQLite applicatif (documents + chat_history)
```

### Règle de dépendance entre modules

```
ui/          → agents/, ingestion/, analytics/, persistence/
agents/      → ingestion/ (embedder), persistence/ (vector_store)
analytics/   → persistence/ (vector_store, metadata_store)
ingestion/   → app/ (config) seulement
persistence/ → app/ (config) seulement
app/         → rien (couche la plus basse)
```

Les `ui/` ne contiennent **que du rendu**. La logique métier est dans `agents/`, `analytics/` et `persistence/`. Un `ui/` ne doit jamais calculer directement un embedding ou appeler ChromaDB — il délègue.

### Où coder quoi ?

| Besoin | Fichier cible |
|--------|---------------|
| Ajouter un nouveau type de fichier (ex: `.pptx`) | `ingestion/loaders.py` |
| Changer le modèle d'embedding | `app/config.py` → `EMBED_MODEL` |
| Changer le LLM de génération | `app/config.py` → `LLM_MODEL` |
| Modifier le prompt RAG | `agents/rag_agent.py` → `_SYSTEM_PROMPT` |
| Ajouter une 6ème visualisation | `analytics/visualizations.py` + `ui/analytics_ui.py` |
| Modifier les stopwords | `analytics/text_stats.py` → `STOPWORDS_FR/EN` |
| Changer la taille des chunks | `app/config.py` → `CHUNK_SIZE` / `CHUNK_OVERLAP` |
| Ajouter un filtre par date dans le chat | `ui/chat_ui.py` + `agents/query_handler.py` |
| Exporter les résultats en CSV | `ui/analytics_ui.py` (ajouter `st.download_button`) |

---

## 5. Roadmap d'implémentation

Cette roadmap reflète l'ordre logique de construction — chaque étape s'appuie sur la précédente.

### Phase 1 — Fondations (fait)

- [x] **Structure du projet** : dossiers, `__init__.py`, `.gitignore`
- [x] **`app/config.py`** : toutes les constantes centralisées
- [x] **`app/state.py`** : gestionnaire du `session_state` Streamlit
- [x] **`requirements.txt`** : dépendances versionnées

### Phase 2 — Pipeline d'ingestion (fait)

- [x] **`ingestion/loaders.py`** : PDF, TXT, DOCX, Markdown
- [x] **`ingestion/scraper.py`** : BeautifulSoup, extraction `<main>`, nettoyage
- [x] **`ingestion/chunker.py`** : stratégie `paragraph` et `sliding_window`
- [x] **`ingestion/embedder.py`** : `OllamaEmbeddings`, batching, vérification connexion
- [x] **`ui/ingestion_ui.py`** : drag-and-drop, aperçu, stats chunks, barre de progression

### Phase 3 — Persistance vectorielle et métadonnées (fait)

- [x] **`persistence/vector_store.py`** : wrapper ChromaDB (`add`, `query`, `delete`, `get_all`)
- [x] **`persistence/metadata_store.py`** : SQLite, tables `documents` + `chat_history`
- [x] Intégration dans le pipeline d'ingestion (remplacement de l'inline ChromaDB)

### Phase 4 — RAG et interface Chat (fait)

- [x] **`agents/rag_agent.py`** : chain LangChain (`embed → retrieve → prompt → LLM`)
- [x] **`agents/query_handler.py`** : validation, gestion erreurs Ollama, historique
- [x] **`ui/chat_ui.py`** : bulles de conversation, panneau de transparence, filtres

### Phase 5 — Tableau de bord analytique (fait)

- [x] **`analytics/text_stats.py`** : fréquences, stopwords intégrés, tokenisation unicode
- [x] **`analytics/topic_model.py`** : BERTopic + UMAP 2D, `partial_fit`, pickle, auto-label
- [x] **`analytics/similarity.py`** : centroïdes, matrice cosinus N×N, numpy pur
- [x] **`analytics/visualizations.py`** : 5 graphiques Plotly + PIL word cloud
- [x] **`ui/analytics_ui.py`** : 5 onglets, cache `session_state`, refresh sélectif

### Phase 6 — Persistance et intelligence incrémentale (fait)

- [x] **`persistence/incremental.py`** : hash SHA-256, déduplication, pipeline post-ingestion
- [x] **Migration SQLite** : `doc_hash` + `ALTER TABLE` automatique
- [x] **Trigger automatique** : `run_incremental_update()` après chaque ingestion réussie
- [x] **Restauration complète** : SQLite + ChromaDB + pickle au redémarrage

### Phase 7 — Améliorations possibles (à venir)

- [ ] **Support OCR** : ajouter `pytesseract` + `pdf2image` pour les PDFs scannés
- [ ] **Export** : bouton "Télécharger en CSV/JSON" sur le dashboard analytique
- [ ] **Historique persistant du chat** : rechargement depuis SQLite au démarrage de session
- [ ] **Recherche hybride** : combiner BM25 (mots-clés) + vecteurs (sémantique) pour le retrieval
- [ ] **Streaming** : utiliser `ChatOllama.stream()` pour afficher la réponse token par token dans le chat
- [ ] **Filtres temporels** : filtrer les analyses par plage de dates d'ingestion
- [ ] **Comparaison de documents** : afficher côte à côte les chunks les plus similaires entre deux docs

---

## Lancement rapide

```bash
# 1. Installer les dépendances Python
pip install -r requirements.txt

# 2. Démarrer Ollama (dans un terminal séparé)
ollama serve

# 3. Télécharger les modèles nécessaires
ollama pull nomic-embed-text   # embeddings (obligatoire)
ollama pull llama3             # LLM de génération (obligatoire)

# 4. Lancer l'application
streamlit run app/main.py
# → ouvre http://localhost:8501
```

**Ordre d'utilisation recommandé :**
1. Page **Ingestion** → ajouter des documents PDF ou des URLs
2. Page **Chat** → poser des questions sur les documents
3. Page **Dashboard** → explorer les visualisations analytiques