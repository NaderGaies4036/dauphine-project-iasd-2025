
import os
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
import requests

# Import LangSmith
try:
    from langsmith import traceable, Client
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
    
    # Initialiser le client LangSmith
    langsmith_client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    )
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith non disponible. Installez-le avec: pip install langsmith")
    langsmith_client = None

# Configuration
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("TELECOMPLUS_LLM_MODEL", "openai/gpt-oss-120b")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
XLSX_DIR = os.path.join(DATA_DIR, "xlsx")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TelecomPlusAgent")

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{6,}")

# Imports optionnels
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    import chromadb
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


@dataclass
class RetrievalResult:
    question: str
    context: str
    source_chunks: List[str]


class TelecomPlusAgent:
    def __init__(self):
        self.vectorstore = None
        self.pdf_chunks: List[str] = []
        self.dataframes: Dict[str, "pd.DataFrame"] = {}
        self._init_data()

    def _init_data(self) -> None:
        self._load_excels()
        self._build_or_load_pdf_index()

    def _load_excels(self) -> None:
        if pd is None:
            logger.warning("pandas non disponible")
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

    def _build_or_load_pdf_index(self) -> None:
        pdf_files = [
            f for f in os.listdir(PDF_DIR)
            if f.lower().endswith(".pdf")
        ] if os.path.isdir(PDF_DIR) else []

        if not pdf_files:
            logger.warning("Aucun PDF trouvé dans %s", PDF_DIR)
            return

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
            logger.warning("Chroma/Embeddings indisponibles, fallback en mémoire.")
            texts = []
            if PyPDFLoader:
                for pdf in pdf_files:
                    loader = PyPDFLoader(os.path.join(PDF_DIR, pdf))
                    docs = loader.load()
                    texts.extend([d.page_content for d in docs])
            self.pdf_chunks = texts

    @traceable(name="retrieve_from_pdfs", run_type="retriever")
    def retrieve_from_pdfs(self, question: str, k: int = 6) -> RetrievalResult:
        """RAG avec monitoring LangSmith"""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(question, k=k)
            chunks = [d.page_content for d in docs]
        else:
            chunks = self.pdf_chunks[:k] if self.pdf_chunks else []

        context = "\n\n---\n\n".join(chunks)
        result = RetrievalResult(
            question=question,
            context=context,
            source_chunks=chunks,
        )
        
        return result

    @traceable(name="get_client_data", run_type="tool")
    def get_client_by_email_or_phone(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Récupération données client avec monitoring"""
        if not identifier or pd is None or "clients" not in self.dataframes:
            return None

        df = self.dataframes["clients"]
        mask = (
            df["email"].astype(str).str.contains(identifier, case=False, na=False)
            | df["telephone"].astype(str).str.contains(identifier, case=False, na=False)
        )
        rows = df[mask]
        
        if rows.empty:
            return None
        
        result = rows.iloc[0].to_dict()
        return result

    def get_client_invoices_summary(self, client_id: Any) -> str:
        if pd is None or "factures" not in self.dataframes:
            return "Les informations de facturation ne sont pas disponibles."

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

    def _needs_client_data(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "ma facture", "mes factures", "mon forfait", "mon abonnement",
            "mes consommations", "mon engagement", "mon contrat", "mon compte",
        ]
        return any(k in q for k in keywords)

    @traceable(name="classify_query", run_type="tool")
    def _classify_query(self, question: str) -> str:
        """Classification avec monitoring"""
        q = question.lower()

        if self._needs_client_data(question):
            query_type = "client"
        else:
            billing_keywords = ["facture", "paiement", "échéance", "prélèvement"]
            tech_keywords = ["réseau", "4g", "5g", "internet", "débit", "panne", "coupure"]
            offer_keywords = ["forfait", "offre", "option", "tarif", "engagement", "roaming"]

            if any(k in q for k in billing_keywords):
                query_type = "billing"
            elif any(k in q for k in tech_keywords):
                query_type = "technical"
            elif any(k in q for k in offer_keywords):
                query_type = "offer"
            else:
                query_type = "generic"
        
        return query_type

    def _extract_identifier(self, question: str) -> str:
        email_match = EMAIL_REGEX.search(question)
        if email_match:
            return email_match.group(0)

        phone_match = PHONE_REGEX.search(question)
        if phone_match:
            phone = re.sub(r"[^\d+]", "", phone_match.group(0))
            return phone

        return ""

    @traceable(name="call_llm", run_type="llm")
    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        """Appel LLM avec monitoring complet"""
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY manquant")
            return "Le service IA externe n'est pas configuré."

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
            answer = data["choices"][0]["message"]["content"]
            
            # Ajouter les métadonnées pour LangSmith
            if LANGSMITH_AVAILABLE:
                try:
                    run_tree = get_current_run_tree()
                    if run_tree:
                        usage = data.get("usage", {})
                        run_tree.extra = {
                            "model": LLM_MODEL,
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                except Exception as e:
                    logger.debug(f"Impossible d'ajouter les métadonnées LangSmith: {e}")
            
            return answer
        except Exception as e:
            logger.error("Erreur appel OpenRouter: %s", e)
            return "Une erreur technique s'est produite."

    @traceable(
        name="answer_customer_query",
        run_type="chain",
        project_name="telecomplus-agent"
    )
    def answer(self, question: str) -> str:
        """Point d'entrée principal avec monitoring complet"""
        query_type = self._classify_query(question)
        needs_data = (query_type == "client")

        # RAG
        retrieval = self.retrieve_from_pdfs(question, k=6)

        # Données client
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
                    "Aucune information client spécifique n'a pu être trouvée."
                )

        # Prompts
        system_prompt = (
            "Tu es un agent de support client expert pour TelecomPlus. "
            "Tu réponds en français, avec un ton professionnel, empathique et rassurant. "
            "Base-toi uniquement sur le contexte fourni (FAQ PDF et données tabulaires). "
            "Tes réponses doivent être factuelles, structurées et centrées sur la résolution."
        )

        user_prompt_parts = [
            f"Question du client :\n{question}",
            f"Type de question détecté : {query_type}",
            "\n===== CONTEXTE FAQ (PDF) =====\n",
            retrieval.context or "(aucun contexte FAQ disponible).",
        ]

        if data_context:
            user_prompt_parts.append("\n===== CONTEXTE DONNÉES CLIENT =====\n")
            user_prompt_parts.append(data_context)

        user_prompt_parts.append(
            "\n===== CONSIGNES =====\n"
            "- Réponds directement et clairement\n"
            "- Personnalise si données client disponibles\n"
            "- Explique les conditions importantes\n"
            "- Propose des démarches concrètes si info manquante"
        )

        user_prompt = "\n".join(user_prompt_parts)
        answer = self._call_openrouter(system_prompt, user_prompt)
        
        # Ajouter métadonnées pour LangSmith
        if LANGSMITH_AVAILABLE:
            try:
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.extra = {
                        "query_type": query_type,
                        "rag_chunks_used": len(retrieval.source_chunks),
                        "client_data_used": bool(data_context),
                        "question_length": len(question),
                    }
            except Exception as e:
                logger.debug(f"Impossible d'ajouter les métadonnées: {e}")
        
        return answer.strip()


_agent: Optional[TelecomPlusAgent] = None


def get_agent() -> TelecomPlusAgent:
    global _agent
    if _agent is None:
        _agent = TelecomPlusAgent()
    return _agent


def answer(question: str) -> str:
    return get_agent().answer(question)