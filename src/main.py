# src/main.py
"""
TelecomPlusAgent - Version optimisée pour évaluation avec OpenRouter (Mixtral / Claude).

Fonctionnalités :
- Charge les PDFs (FAQ) depuis data/pdfs et les Excels depuis data/xlsx
- Indexe les PDFs avec Chroma si disponible, sinon garde le texte en mémoire
- Fournit des outils "SQL-like" sur les fichiers Excel via pandas
- Orchestration entre RAG (FAQ) et données clients (Excel)
- Extraction d'email / téléphone (et extensible au nom) depuis la question
- Appelle OpenRouter (modèle configurable) comme générateur final
- Expose `answer(question)` compatible avec evaluate.py

Dépendances :
pip install python-dotenv requests pandas PyPDF2
Optionnel (pour Chroma/embeddings) :
pip install chromadb langchain langchain-community langchain-text-splitters tf-keras
"""

import os
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
import requests

# --------------------------------------------------------------------------- #
# Configuration globale
# --------------------------------------------------------------------------- #

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Tu peux changer le modèle ici ou via TELECOMPLUS_LLM_MODEL dans ton .env
LLM_MODEL = os.getenv("TELECOMPLUS_LLM_MODEL", "anthropic/claude-3.7-sonnet")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
XLSX_DIR = os.path.join(DATA_DIR, "xlsx")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TelecomPlusAgent")

# Regex simples pour email et téléphone
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{6,}")

# --------------------------------------------------------------------------- #
# Dépendances optionnelles (RAG / vecteurs / LLM)
# --------------------------------------------------------------------------- #
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("Imports LangChain / Chroma OK.")
except Exception as e:
    logger.error("Erreur import LangChain / Chroma: %s", e)
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    Chroma = None
    HuggingFaceEmbeddings = None

try:
    import pandas as pd
except Exception:
    pd = None


# --------------------------------------------------------------------------- #
# Structures de données
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalResult:
    question: str
    context: str
    source_chunks: List[str]


class TelecomPlusAgent:
    """
    Agent principal :
    - RAG sur les FAQ PDF
    - Outils tabulaires (Excel) pour les données clients
    - Génération finale via OpenRouter
    """

    def __init__(self):
        # PDF / RAG
        self.vectorstore = None
        self.pdf_chunks: List[str] = []

        # Données tabulaires
        self.dataframes: Dict[str, "pd.DataFrame"] = {}

        # Initialisation des données
        self._init_data()

    # ------------------------------------------------------------------ #
    # Initialisation des données
    # ------------------------------------------------------------------ #
    def _init_data(self) -> None:
        self._load_excels()
        self._build_or_load_pdf_index()

    def _load_excels(self) -> None:
        if pd is None:
            logger.warning("pandas non disponible : les outils tabulaires sont désactivés.")
            return

        filenames = [
            "clients.xlsx",
            "forfaits.xlsx",
            "abonnements.xlsx",
            "consommation.xlsx",
            "factures.xlsx",
            "tickets_support.xlsx",
        ]
        for name in filenames:
            path = os.path.join(XLSX_DIR, name)
            if os.path.exists(path):
                try:
                    self.dataframes[name.replace(".xlsx", "")] = pd.read_excel(path)
                    logger.info("Chargé: %s", path)
                except Exception as e:
                    logger.error("Erreur de chargement %s: %s", path, e)
            else:
                logger.warning("Fichier Excel introuvable: %s", path)

    def _build_or_load_pdf_index(self) -> None:
        pdf_files = [
            f for f in os.listdir(PDF_DIR)
            if f.lower().endswith(".pdf")
        ] if os.path.isdir(PDF_DIR) else []

        if not pdf_files:
            logger.warning("Aucun PDF trouvé dans %s", PDF_DIR)
            return

        # Si Chroma + embeddings disponibles, on construit un index
        if PyPDFLoader and RecursiveCharacterTextSplitter and Chroma and HuggingFaceEmbeddings:
            logger.info("Construction de l'index Chroma pour les FAQ...")
            docs = []
            for pdf in pdf_files:
                loader = PyPDFLoader(os.path.join(PDF_DIR, pdf))
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150,
            )
            split_docs = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings()
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                collection_name="telecomplus_faq",
            )
            logger.info("Index Chroma construit (%d chunks).", len(split_docs))
        else:
            # Fallback : garder seulement le texte en mémoire
            logger.warning("Chroma/Embeddings indisponibles, fallback en mémoire.")
            texts = []
            if not PyPDFLoader:
                logger.warning("PyPDFLoader indisponible, impossible de charger les PDF.")
            else:
                for pdf in pdf_files:
                    loader = PyPDFLoader(os.path.join(PDF_DIR, pdf))
                    docs = loader.load()
                    texts.extend([d.page_content for d in docs])
            self.pdf_chunks = texts

    # ------------------------------------------------------------------ #
    # RAG PDF
    # ------------------------------------------------------------------ #
    def retrieve_from_pdfs(self, question: str, k: int = 6) -> RetrievalResult:
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(question, k=k)
            chunks = [d.page_content for d in docs]
        else:
            chunks = self.pdf_chunks[:k] if self.pdf_chunks else []

        context = "\n\n---\n\n".join(chunks)
        return RetrievalResult(
            question=question,
            context=context,
            source_chunks=chunks,
        )

    # ------------------------------------------------------------------ #
    # Outils "SQL-like" sur les Excels (simplifiés)
    # ------------------------------------------------------------------ #
    def get_client_by_email_or_phone(self, identifier: str) -> Optional[Dict[str, Any]]:
        if not identifier:
            return None
        if pd is None or "clients" not in self.dataframes:
            return None

        df = self.dataframes["clients"]
        mask = (
            df["email"].astype(str).str.contains(identifier, case=False, na=False)
            | df["telephone"].astype(str).str.contains(identifier, case=False, na=False)
        )
        rows = df[mask]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def get_client_invoices_summary(self, client_id: Any) -> str:
        if pd is None or "factures" not in self.dataframes:
            return "Les informations de facturation ne sont pas disponibles pour le moment."

        df = self.dataframes["factures"]
        rows = df[df["client_id"] == client_id]
        if rows.empty:
            return "Aucune facture trouvée pour ce client."
        latest = rows.sort_values("date_echeance").iloc[-1]
        return (
            f"Dernière facture : montant {latest.get('montant')} €, "
            f"statut {latest.get('statut_paiement')}, "
            f"échéance le {latest.get('date_echeance')}."
        )

    # ------------------------------------------------------------------ #
    # Routing / classification de la question
    # ------------------------------------------------------------------ #
    def _needs_client_data(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "ma facture",
            "mes factures",
            "mon forfait",
            "mon abonnement",
            "mes consommations",
            "mon engagement",
            "mon contrat",
            "mon compte",
        ]
        return any(k in q for k in keywords)

    def _classify_query(self, question: str) -> str:
        q = question.lower()

        if self._needs_client_data(question):
            return "client"

        billing_keywords = ["facture", "paiement", "échéance", "prélèvement"]
        tech_keywords = ["réseau", "4g", "5g", "internet", "débit", "panne", "coupure"]
        offer_keywords = ["forfait", "offre", "option", "tarif", "engagement", "roaming", "international"]

        if any(k in q for k in billing_keywords):
            return "billing"
        if any(k in q for k in tech_keywords):
            return "technical"
        if any(k in q for k in offer_keywords):
            return "offer"

        return "generic"

    # ------------------------------------------------------------------ #
    # Extraction d'identifiant client (email / téléphone)
    # ------------------------------------------------------------------ #
    def _extract_identifier(self, question: str) -> str:
        # Email en priorité
        email_match = EMAIL_REGEX.search(question)
        if email_match:
            return email_match.group(0)

        # Téléphone simple
        phone_match = PHONE_REGEX.search(question)
        if phone_match:
            phone = re.sub(r"[^\d+]", "", phone_match.group(0))
            return phone

        return ""

    # ------------------------------------------------------------------ #
    # Appel OpenRouter
    # ------------------------------------------------------------------ #
    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY manquant, réponse de fallback.")
            return "Le service IA externe n'est pas configuré. Merci de réessayer plus tard."

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Erreur appel OpenRouter: %s", e)
            return "Une erreur technique s'est produite lors de la génération de la réponse."

    # ------------------------------------------------------------------ #
    # Orchestration principale
    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> str:
        """
        Point d'entrée principal, utilisé par evaluate.py et app.py.
        """
        query_type = self._classify_query(question)
        needs_data = (query_type == "client")

        # 1) RAG PDF
        retrieval = self.retrieve_from_pdfs(question, k=6)

        # 2) Données client (si pertinent)
        data_context = ""
        if needs_data:
            identifier = self._extract_identifier(question)
            client = self.get_client_by_email_or_phone(identifier) if identifier else None

            if client:
                client_id = client.get("id")
                factures_summary = self.get_client_invoices_summary(client_id)
                data_context = (
                    "Informations client:\n"
                    + json.dumps(client, default=str, ensure_ascii=False)
                    + "\n\nRésumé factures:\n"
                    + factures_summary
                )
            else:
                data_context = (
                    "Aucune information client spécifique n'a pu être trouvée "
                    "(identifiant client non présent ou non reconnu dans la question)."
                )

        # 3) Prompt système
        system_prompt = (
            "Tu es un agent de support client expert pour un opérateur téléphonique fictif nommé TelecomPlus. "
            "Tu réponds en français, avec un ton professionnel, empathique et rassurant. "
            "Tu aides les clients sur des sujets comme : factures, forfaits, options, consommation, "
            "problèmes de réseau, roaming et support technique. "
            "Base-toi uniquement sur le contexte fourni (FAQ PDF et données tabulaires Excel). "
            "Ne fais pas de suppositions sur des données clients si elles ne sont pas explicitement présentes. "
            "Quand une information manque, indique-le clairement et propose des étapes concrètes "
            '(par exemple : \"connectez-vous à votre espace client\" ou \"contactez le service client au numéro indiqué sur votre facture\"). '
            "Tes réponses doivent être factuelles, structurées et centrées sur la résolution du problème du client."
        )

        # 4) Prompt utilisateur structuré
        user_prompt_parts = [
            "Tu vas répondre à un client de TelecomPlus.",
            f"Question du client :\n{question}",
            f"Type de question détecté : {query_type}",
            "\n===== CONTEXTE FAQ (PDF) =====\n",
            retrieval.context or "(aucun contexte FAQ disponible).",
        ]

        if data_context:
            user_prompt_parts.append("\n===== CONTEXTE DONNÉES CLIENT (Excel) =====\n")
            user_prompt_parts.append(data_context)

        user_prompt_parts.append(
            "\n===== CONSIGNES DE RÉPONSE =====\n"
            "- Commence par une phrase courte qui répond directement à la question du client.\n"
            "- Ensuite, détaille les informations importantes (montants, dates, conditions, démarches) en quelques phrases ou points.\n"
            "- Si des informations clients sont disponibles, personnalise la réponse (par exemple : montant, échéance, statut de facture).\n"
            "- Si des informations sont manquantes, explique-le poliment et propose des démarches concrètes.\n"
            "- Si la question concerne un forfait ou une offre, explique clairement les conditions importantes (prix, durée d'engagement, options, limitations) présentes dans le contexte.\n"
            "- Ne mentionne pas ce contexte interne, ni les noms de fichiers, ni les mots 'FAQ', 'Excel' ou 'PDF'.\n"
            "- Ne génère pas d'informations qui ne figurent pas dans le contexte.\n"
            "- Formule une seule réponse finale en français, adaptée au client."
        )

        user_prompt = "\n".join(user_prompt_parts)

        # 5) Appel au modèle
        answer = self._call_openrouter(system_prompt, user_prompt)
        return answer.strip()


# --------------------------------------------------------------------------- #
# Instance globale + wrapper pour evaluate.py
# --------------------------------------------------------------------------- #

_agent: Optional[TelecomPlusAgent] = None


def get_agent() -> TelecomPlusAgent:
    global _agent
    if _agent is None:
        _agent = TelecomPlusAgent()
    return _agent


def answer(question: str) -> str:
    """
    Wrapper simple à utiliser dans evaluate.py :
    from src.main import answer
    resp = answer("Ma question")
    """
    return get_agent().answer(question)
