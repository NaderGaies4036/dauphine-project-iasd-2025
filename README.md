# 📱 TelecomPlus - Agent IA de Support Client

Agent intelligent de support client pour opérateur télécom utilisant RAG (Retrieval Augmented Generation), requêtes sur données structurées et monitoring avancé.

---

## 🎯 Fonctionnalités

### ✅ Agent IA Multi-sources

1. **RAG sur documents PDF**
   - Indexation vectorielle avec ChromaDB
   - Embeddings HuggingFace (sentence-transformers)
   - Recherche sémantique dans la FAQ
   - Fallback en mémoire si ChromaDB indisponible

2. **Requêtes sur données structurées**
   - 6 tables Excel : clients, forfaits, abonnements, consommation, factures, tickets
   - Requêtes pandas pour extraction d'informations clients
   - Jointures automatiques entre tables

3. **Orchestration intelligente**
   - Classification automatique des questions (client, billing, technical, offer, generic)
   - Extraction d'identifiants (email, téléphone)
   - Routage vers les sources appropriées
   - Génération de réponses contextualisées

### 📊 Évaluation automatique

- Script d'évaluation sur 25 questions de test
- LLM-as-a-judge 
- Scoring de 0 à 1 avec justifications
- Export des résultats dans Excel

Le score obtenu est de 0.8
### 🔍 Monitoring & Observabilité

- Intégration Langfuse pour traçage complet
- Métriques de performance (latence, tokens)
- Visualisation des étapes RAG et LLM
- Dashboard de suivi en temps réel

### 🖥️ Interface utilisateur

- Chat Streamlit interactif
- Historique de conversation
- Interface en français

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Question Client                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Classification & Extraction                   │
│  • Type de question (client/billing/technical/offer)    │
│  • Identifiants (email/téléphone)                       │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│                  Orchestration                          │
├─────────────────────┬──────────────────────────────────┤
│   RAG (PDF)         │   Données Client (Excel)         │
│ • Recherche FAQ     │ • Info client                    │
│ • Top-6 chunks      │ • Factures                       │
│ • Context enrichi   │ • Abonnements                    │
└─────────────────────┴──────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              Génération LLM (OpenRouter)                │
│           • openai/gpt-oss-120b                 │
│           • Prompt enrichi avec contexte                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Réponse Personnalisée                      │
└─────────────────────────────────────────────────────────┘
```

---

#### 📍 Traces
Vue complète de chaque requête client :
```
telecomplus_query
├── classify_query        (type de question)
├── retrieve_from_pdfs    (RAG - 6 chunks)
├── get_client_data       (données Excel)
├── call_llm              (génération réponse)
└── answer_customer_query (réponse finale)
```

## 🛠️ Technologies

### Backend & LLM
- **LangChain** : Framework pour applications LLM
- **OpenRouter** : API unifiée pour accès aux LLMs (Claude, GPT, Mixtral, etc.)
- **openai/gpt-oss-120b** : Modèle de génération par défaut

### RAG & Embeddings
- **ChromaDB** : Base de données vectorielle
- **Sentence Transformers** : Modèles d'embeddings open-source
- **PyPDF2** : Extraction de texte des PDFs

### Données structurées
- **Pandas** : Manipulation et requêtes sur données tabulaires
- **OpenPyXL** : Lecture/écriture de fichiers Excel

### Monitoring
- **Langfuse** : Plateforme de monitoring et debugging pour LLM
  - Tracing distribué
  - Analyse de performance
  - Gestion des coûts
  - Debugging conversationnel

### Interface utilisateur
- **Streamlit** : Framework pour applications web Python

---

## 💻 Utilisation

### Option 1 : Interface Streamlit


```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

**Fonctionnalités :**
- Chat en temps réel
- Historique de conversation
- Réponses contextualisées
- Support FAQ + données client

**Exemples de questions :**

```
💬 Questions générales (FAQ) :
- "Quels sont vos forfaits disponibles ?"
- "Comment activer le roaming international ?"
- "Que faire en cas de panne réseau ?"

💬 Questions personnalisées (avec email/téléphone) :
- "Quel est le montant de ma facture ? (client@example.com)"
- "Ma consommation ce mois-ci ? (+33612345678)"
- "Quand expire mon engagement ? (client@example.com)"
```

---

## 📁 Structure du projet

```
dauphine-project-iasd-2025/
│
├── 📂 data/                        # Données du projet
│   ├── 📂 pdfs/                    # Documents FAQ (PDF)
│   │   └── faq_telecomplus.pdf
│   ├── 📂 xlsx/                    # Données structurées (Excel)
│   │   ├── clients.xlsx            # Informations clients
│   │   ├── forfaits.xlsx           # Catalogue forfaits
│   │   ├── abonnements.xlsx        # Abonnements actifs
│   │   ├── consommation.xlsx       # Historique consommation
│   │   ├── factures.xlsx           # Factures et paiements
│   │   └── tickets_support.xlsx    # Tickets support
│   └── evaluation_questions.xlsx   # Questions de test (25)
│
├── 🤖 main.py                      # Agent principal avec LangSmith
├── 💬 app.py                       # Interface Streamlit
├── 📊 evaluate.py                  # Script d'évaluation
│
├── 🔧 requirements.txt             # Dépendances Python
├── 🔑 .env                         # Variables d'environnement (à créer)
├── 📖 README.md                    # Documentation (ce fichier)
│
└── 📄 evaluation_results.xlsx      # Résultats d'évaluation (généré)
```

---

## 🛠️ Technologies

### Backend & LLM

| Technologie | Utilisation | Version |
|-------------|-------------|---------|
| **Python** | Langage principal | 3.9+ |
| **LangChain** | Framework LLM | 0.1.0 |
| **OpenRouter** | API LLM unifiée | - |
| **openai/gpt-oss-120b** | Modèle de génération | Latest |

### RAG & Embeddings

| Technologie | Utilisation | Version |
|-------------|-------------|---------|
| **ChromaDB** | Base vectorielle | 0.4.22 |
| **Sentence Transformers** | Embeddings | 2.2.2 |
| **PyPDF2** | Extraction PDF | 3.0.1 |

### Données

| Technologie | Utilisation | Version |
|-------------|-------------|---------|
| **Pandas** | Manipulation données | 2.1.4 |
| **OpenPyXL** | Lecture/écriture Excel | 3.1.2 |

### Monitoring

| Technologie | Utilisation | Version |
|-------------|-------------|---------|
| **LangSmith** | Tracing & monitoring | 0.0.87 |

### Interface

| Technologie | Utilisation | Version |
|-------------|-------------|---------|
| **Streamlit** | Interface web | 1.29.0 |

---