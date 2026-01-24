import os
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
import requests
from langsmith import get_current_run_tree
# Configuration
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("TELECOMPLUS_LLM_MODEL", "openai/gpt-oss-120b")

# Import LangSmith APRÈS load_dotenv()
try:
    from langsmith import traceable
    # Vérifier que les variables sont bien définies
    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING") == "true":
        LANGSMITH_AVAILABLE = True
        logging.info("LangSmith activé avec succès")
    else:
        LANGSMITH_AVAILABLE = False
        logging.warning("LangSmith désactivé : variables d'environnement manquantes")
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith non disponible. Installez-le avec: pip install langsmith")
    
    # Créer un décorateur dummy si LangSmith n'est pas disponible
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
XLSX_DIR = os.path.join(DATA_DIR, "xlsx")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TelecomPlusAgent")

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{6,}")
NAME_PATTERN = re.compile(r"(?:je m'appelle|je suis|mon nom est|c'est)\s+([A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+))", re.IGNORECASE)

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
    def get_client_by_identifier(self, identifier: str, identifier_type: str = "auto") -> Optional[Dict[str, Any]]:

        if not identifier or pd is None or "clients" not in self.dataframes:
            logger.warning("Impossible de rechercher le client : données manquantes")
            return None

        df = self.dataframes["clients"]
        logger.info(f"Recherche client avec '{identifier}' (type: {identifier_type})")
        logger.info(f"Colonnes disponibles : {list(df.columns)}")
        
        if identifier_type == "name":
            # Normaliser le nom pour la recherche (minuscules, sans accents extrêmes)
            identifier_lower = identifier.lower()
            parts = identifier_lower.split()
            
            # Recherche flexible : nom OU prénom OU nom complet
            mask = pd.Series([False] * len(df))
            
            # Vérifier chaque partie du nom
            for part in parts:
                mask = mask | df["nom"].astype(str).str.lower().str.contains(part, na=False)
                mask = mask | df["prenom"].astype(str).str.lower().str.contains(part, na=False)
            
            # Vérifier aussi le nom complet
            full_name_col = (df["prenom"].astype(str) + " " + df["nom"].astype(str)).str.lower()
            mask = mask | full_name_col.str.contains(identifier_lower, na=False)
            
            #print(f"Résultats pour recherche par nom : {mask.sum()} client(s)")
            
        elif identifier_type == "email":
            mask = df["email"].astype(str).str.lower().str.contains(identifier.lower(), case=False, na=False)
            logger.info(f"Résultats pour recherche par email : {mask.sum()} client(s)")
            
        elif identifier_type == "phone":
            # Nettoyer le téléphone pour la comparaison
            clean_phone = re.sub(r'[^\d+]', '', identifier)
            mask = df["telephone"].astype(str).str.replace(r'[^\d+]', '', regex=True).str.contains(clean_phone, na=False)
            logger.info(f"Résultats pour recherche par téléphone : {mask.sum()} client(s)")
            
        else:  # auto
            identifier_lower = identifier.lower()
            mask = (
                df["email"].astype(str).str.lower().str.contains(identifier_lower, na=False) |
                df["telephone"].astype(str).str.contains(identifier, na=False) |
                df["nom"].astype(str).str.lower().str.contains(identifier_lower, na=False) |
                df["prenom"].astype(str).str.lower().str.contains(identifier_lower, na=False)
            )
            #print(f"Résultats pour recherche auto : {mask.sum()} client(s)")
        
        rows = df[mask]
        
        if rows.empty:
            #print(f"Aucun client trouvé pour '{identifier}'")
            if identifier_type == "name":
                all_names = [f"{row['prenom']} {row['nom']}" for _, row in df.iterrows()]
                #print(f"Noms disponibles dans la base : {all_names[:5]}...")
            return None
        
        result = rows.iloc[0].to_dict()
        #print(f"✅ Client trouvé : {result.get('prenom')} {result.get('nom')} (ID: {result.get('client_id')})")
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
    def get_client_subscription_info(self, client_id: Any) -> str:
        """Récupère les informations d'abonnement du client"""
        if pd is None or "abonnements" not in self.dataframes:
            return ""
        
        df = self.dataframes["abonnements"]
        rows = df[df["client_id"] == client_id]
        if rows.empty:
            return ""
        
        sub = rows.iloc[0]
        forfait_id = sub.get("forfait_id")
        
        # Récupérer les détails du forfait
        forfait_info = ""
        if "forfaits" in self.dataframes:
            forfait_df = self.dataframes["forfaits"]
            forfait_rows = forfait_df[forfait_df["forfait_id"] == forfait_id]
            if not forfait_rows.empty:
                forfait = forfait_rows.iloc[0]
                forfait_info = (
                    f"Forfait actuel : {forfait.get('nom_forfait', 'N/A')}\n"
                    f"Prix : {forfait.get('prix_mensuel', 'N/A')} €/mois\n"
                    f"Data : {forfait.get('data_mensuel_gb', 'N/A')} Go\n"
                    f"Appels : {forfait.get('minutes_incluses', 'N/A')}\n"
                    f"SMS : {forfait.get('sms_inclus', 'N/A')}\n"
                )
        
        return (
            f"{forfait_info}"
            f"Date de début : {sub.get('date_debut', 'N/A')}\n"
            f"Date de fin d'engagement : {sub.get('date_fin', 'N/A')}\n"
            f"Statut : {sub.get('statut', 'N/A')}"
        )

    def get_client_consumption(self, client_id: Any) -> str:
        """Récupère la consommation du client"""
        if pd is None or "consommation" not in self.dataframes:
            return ""
        
        df = self.dataframes["consommation"]
        rows = df[df["client_id"] == client_id]
        if rows.empty:
            return ""
        
        latest = rows.sort_values("mois", ascending=False).iloc[0]
        return (
            f"Consommation du mois de {latest.get('mois', 'N/A')} :\n"
            f"- Data utilisée : {latest.get('data_utilise_gb', 0)} Go\n"
            f"- Appels : {latest.get('minutes_utilisees', 0)} minutes\n"
            f"- SMS envoyés : {latest.get('sms_utilises', 0)}"
        )
    def get_client_tickets(self, client_id: Any) -> str:
        """Récupère les tickets de support du client"""
        if pd is None or "tickets_support" not in self.dataframes:
            return ""
        
        df = self.dataframes["tickets_support"]
        rows = df[df["client_id"] == client_id]
        if rows.empty:
            return "Aucun ticket de support en cours."
        
        # Filtrer les tickets en cours (statut != 'Résolu')
        open_tickets = rows[rows["statut"] != "Résolu"]
        
        if open_tickets.empty:
            return "Aucun ticket de support en cours. Tous vos tickets ont été résolus."
        
        ticket_parts = []
        for idx, ticket in open_tickets.iterrows():
            ticket_parts.append(
                f"Ticket #{ticket.get('ticket_id')} : '{ticket.get('sujet')}' "
                f"(Catégorie : {ticket.get('categorie')}, Statut : {ticket.get('statut')})"
            )
        
        return f"Vous avez {len(open_tickets)} ticket(s) en cours :\n" + "\n".join(ticket_parts)


    def _needs_client_data(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "ma facture", "mes factures", "mon forfait", "mon abonnement",
            "mes consommations", "mon engagement", "mon contrat", "mon compte",
        ]
        return any(k in q for k in keywords)

    @traceable(name="classify_query", run_type="tool")
    def _classify_query(self, question: str) -> str:
        """Classification de la question"""
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

    def _extract_identifier(self, question: str) -> Tuple[str, str]:
        """
        Extrait l'identifiant client et son type
        Returns: (identifier, type) où type in ['email', 'phone', 'name', '']
        """
        # 1. Chercher un nom (priorité haute pour les questions personnelles)
        name_match = NAME_PATTERN.search(question)
        if name_match:
            full_name = name_match.group(1).strip()
            logger.info(f"Nom extrait: {full_name}")
            return (full_name, "name")
        
        # 2. Chercher un email
        email_match = EMAIL_REGEX.search(question)
        if email_match:
            return (email_match.group(0), "email")

        # 3. Chercher un téléphone
        phone_match = PHONE_REGEX.search(question)
        if phone_match:
            phone = re.sub(r"[^\d+]", "", phone_match.group(0))
            return (phone, "phone")

        return ("", "")

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
        client_found = False

        if needs_data or "je m'appelle" in question.lower() or "je suis" in question.lower():
            identifier, id_type = self._extract_identifier(question)
            logger.info(f"Identifiant extrait: '{identifier}' (type: {id_type})")
            
            client = self.get_client_by_identifier(identifier, id_type) if identifier else None

            if client:
                client_found = True
                client_id = client.get("client_id")
                
                # Collecter toutes les informations pertinentes
                context_parts = ["=== INFORMATIONS CLIENT DISPONIBLES ==="]
                context_parts.append(f"Nom : {client.get('nom', 'N/A')} {client.get('prenom', 'N/A')}")
                context_parts.append(f"Email : {client.get('email', 'N/A')}")
                context_parts.append(f"Téléphone : {client.get('telephone', 'N/A')}")
                context_parts.append(f"Statut du compte : {client.get('statut', 'N/A')}")
                
                # Abonnement
                sub_info = self.get_client_subscription_info(client_id)
                if sub_info:
                    context_parts.append("\n=== ABONNEMENT ET FORFAIT ===")
                    context_parts.append(sub_info)
                
                # Consommation
                consumption = self.get_client_consumption(client_id)
                if consumption:
                    context_parts.append("\n=== CONSOMMATION ACTUELLE ===")
                    context_parts.append(consumption)
                
                # Factures
                factures_summary = self.get_client_invoices_summary(client_id)
                context_parts.append("\n=== FACTURES RÉCENTES ===")
                context_parts.append(factures_summary)
                
                # Tickets
                tickets = self.get_client_tickets(client_id)
                if tickets:
                    context_parts.append("\n=== TICKETS DE SUPPORT ===")
                    context_parts.append(tickets)
                
                data_context = "\n".join(context_parts)
            else:
                if identifier:
                    data_context = (
                        f"⚠️ AUCUNE DONNÉE CLIENT TROUVÉE pour l'identifiant '{identifier}' (type: {id_type}). "
                        "Cela peut signifier que le client n'existe pas dans la base ou que l'identifiant est incorrect."
                    )

        # Prompts
        system_prompt = ("""Tu es un conseiller clientèle expert de TelecomPlus, opérateur de télécommunications français. 
RÈGLE D'OR ABSOLUE :
Si des DONNÉES CLIENT sont fournies dans le contexte, tu DOIS les utiliser pour répondre de manière PERSONNALISÉE et PRÉCISE.
Tu NE DOIS JAMAIS dire "je ne peux pas accéder" ou "consultez votre espace client" si les données sont dans le contexte.

PRINCIPES FONDAMENTAUX :
1. DONNÉES AVANT TOUT : Si le contexte contient des données client (factures, consommation, forfait), utilise-les DIRECTEMENT dans ta réponse
2. PRÉCISION MAXIMALE : Cite les montants exacts, dates précises, et chiffres spécifiques issus du contexte
3. PERSONNALISATION OBLIGATOIRE : Adresse-toi au client par son nom si disponible
4. ZÉRO HALLUCINATION : N'invente JAMAIS de données. Si absent du contexte, dis-le clairement
        "5. Pour les questions sur les iPhones (prix, coloris, spécifications techniques):\n"
        "   - Si l'information est dans le contexte: utilise-la\n"
        "   - Si l'information N'EST PAS dans le contexte: utilise tes connaissances générales sur les produits Apple et le marché télécom\n"
        "   - Fournis des réponses précises basées sur les standards du marché\n"
6. RÉPONSE DIRECTE : Va droit au but avec l'information demandée

INTERDICTIONS ABSOLUES :
Ne JAMAIS dire "consultez votre espace client" si les données sont dans le contexte
 Ne JAMAIS dire "je ne peux pas accéder à vos données" si elles sont fournies
 Ne JAMAIS inventer des montants, dates ou informations
 Ne JAMAIS être vague si des données précises sont disponibles
 Ne JAMAIS ignorer les données client fournies

FORMAT DE RÉPONSE QUAND DONNÉES CLIENT DISPONIBLES :
1. Salue le client par son nom
2. RÉPONDS DIRECTEMENT avec les chiffres/dates exacts du contexte
3. Explique brièvement si nécessaire
4. Propose une action concrète si pertinent

FORMAT QUAND DONNÉES MANQUANTES :
1. Indique clairement quelle information manque
2. Explique comment le client peut l'obtenir
3. Propose des alternatives basées sur la FAQ

STYLE :
- Professionnel mais chaleureux
- Paragraphes courts (2-3 phrases max)
- Pas de listes à puces sauf si vraiment nécessaire
- Empathique et orienté solution """
        )

        user_prompt_parts = [
            "=== QUESTION DU CLIENT ===",
            question,
            "",
            f"=== TYPE DE DEMANDE : {query_type.upper()} ===",
            ""
        ]

        if data_context:
            user_prompt_parts.extend([
                data_context,
                ""
            ])
            
            if client_found:
                user_prompt_parts.extend([
                    "INSTRUCTION CRITIQUE :",
                    "Les données ci-dessus contiennent TOUTES les informations nécessaires pour répondre.",
                    "Tu DOIS utiliser ces données exactes dans ta réponse.",
                    "N'invite PAS le client à consulter son espace - RÉPONDS DIRECTEMENT avec ces données.",
                    ""
                ])

        if retrieval.context:
            user_prompt_parts.extend([
                "=== BASE DE CONNAISSANCES (FAQ) ===",
                retrieval.context,
                ""
            ])
        response_instructions = {
            "client": """
            INSTRUCTIONS DE RÉPONSE :
            1. Utilise le NOM du client pour le saluer
            2. CITE LES CHIFFRES EXACTS des données (montant, date, Gb utilisés, etc.)
            3. Réponds de manière DIRECTE - pas de détours
            4. Si la donnée est dans le contexte, ne suggère PAS de consulter l'espace client
            5. Termine par une question ou proposition d'aide si pertinent
            - Pour les produits/iPhones: utilise le contexte OU tes connaissances si le contexte est insuffisant
            - Pour les données clients: utilise UNIQUEMENT les données fournies


            """,
            "billing": "Explique clairement avec les tarifs et dates du contexte FAQ. Sois précis sur les modalités.",
            "technical": "Fournis des étapes de dépannage concrètes et numérotées. Rassure le client.",
            "offer": "Compare les forfaits avec le forfait actuel du client si disponible. Indique prix et conditions.",
            "generic": "Réponds avec les informations de la FAQ. Reste professionnel et utile."
        }

        user_prompt_parts.extend([
            response_instructions.get(query_type, response_instructions["generic"]),
            "",
            "RÉPONDS MAINTENANT de manière DIRECTE et PERSONNALISÉE :"
        ])
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