# 📱 TelecomPlus - Agent IA de Support Client

Agent intelligent de support client pour opérateur télécom utilisant RAG (Retrieval Augmented Generation), requêtes sur données structurées et monitoring avancé.

---

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Évaluation](#-évaluation)
- [Monitoring](#-monitoring)
- [Structure du projet](#-structure-du-projet)
- [Technologies](#-technologies)

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
- LLM-as-a-judge (Claude 3.7 Sonnet via OpenRouter)
- Scoring de 0 à 1 avec justifications
- Export des résultats dans Excel

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
│           • Claude 3.7 Sonnet (défaut)                  │
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
- **Claude 3.7 Sonnet** : Modèle de génération par défaut

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