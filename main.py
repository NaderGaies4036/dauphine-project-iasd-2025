import os
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
import requests

# Import Langfuse
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse non disponible. Installez-le avec: pip install langfuse")

# Configuration
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("TELECOMPLUS_LLM_MODEL", "anthropic/claude-3.7-sonnet")

# Initialiser Langfuse si disponible
if LANGFUSE_AVAILABLE:
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
else:
    langfuse = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
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

    @observe(name="retrieve_from_pdfs")
    def retrieve_from_pdfs(self, question: str, k: int = 6) -> RetrievalResult:
        """RAG avec monitoring Langfuse"""
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                input=question,
                metadata={"k": k, "has_vectorstore": self.vectorstore is not None}
            )
        
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
        
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                output={"num_chunks": len(chunks), "context_length": len(context)}
            )
        
        return result

    @observe(name="get_client_data")
    def get_client_by_email_or_phone(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Récupération données client avec monitoring"""
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                input={"identifier_provided": bool(identifier)}
            )
        
        if not identifier or pd is None or "clients" not in self.dataframes:
            return None

        df = self.dataframes["clients"]
        mask = (
            df["email"].astype(str).str.contains(identifier, case=False, na=False)
            | df["telephone"].astype(str).str.contains(identifier, case=False, na=False)
        )
        rows = df[mask]
        
        if rows.empty:
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_observation(
                    output={"client_found": False}
                )
            return None
        
        result = rows.iloc[0].to_dict()
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                output={"client_found": True, "client_id": result.get("id")}
            )
        
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

    @observe(name="classify_query")
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
        
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                input=question,
                output={"query_type": query_type}
            )
        
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

    @observe(name="call_llm")
    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        """Appel LLM avec monitoring complet"""
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                model=LLM_MODEL,
                input={"system": system_prompt, "user": user_prompt},
                metadata={"provider": "openrouter"}
            )
        
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
            
            if LANGFUSE_AVAILABLE:
                usage = data.get("usage", {})
                langfuse_context.update_current_observation(
                    output=answer,
                    usage={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0)
                    }
                )
            
            return answer
        except Exception as e:
            logger.error("Erreur appel OpenRouter: %s", e)
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )
            return "Une erreur technique s'est produite."

    @observe(name="answer_customer_query")
    def answer(self, question: str) -> str:
        """Point d'entrée principal avec monitoring complet"""
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_trace(
                name="telecomplus_query",
                user_id="evaluation_script",
                metadata={"question_length": len(question)}
            )
            langfuse_context.update_current_observation(
                input=question
            )
        
        query_type = self._classify_query(question)
        needs_data = (query_type == "client")

        retrieval = self.retrieve_from_pdfs(question, k=6)

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
        
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_observation(
                output=answer,
                metadata={
                    "query_type": query_type,
                    "rag_chunks_used": len(retrieval.source_chunks),
                    "client_data_used": bool(data_context)
                }
            )
        
        return answer.strip()


_agent: Optional[TelecomPlusAgent] = None

def get_agent() -> TelecomPlusAgent:
    global _agent
    if _agent is None:
        _agent = TelecomPlusAgent()
    return _agent

def answer(question: str) -> str:
    return get_agent().answer(question)