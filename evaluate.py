import os
import json
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    tqdm = None

from src.main import answer as agent_answer


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_PATH = os.path.join(DATA_DIR, "evaluation_questions.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation_results.xlsx")

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TelecomPlusEvaluation")


def _build_openrouter_payload(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    return {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
    }


def _post_openrouter(body: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY manquant")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _safe_parse_json_content(content: str) -> Optional[Dict[str, Any]]:
    """
    Essaye de parser un JSON même si le modèle ajoute du texte autour.
    On cherche le premier bloc {...} valide.
    """
    content = content.strip()
    # Tentative directe
    try:
        return json.loads(content)
    except Exception:
        pass

    # Fallback: chercher le premier '{' et le dernier '}'
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        snippet = content[start:end]
        return json.loads(snippet)
    except Exception:
        return None


def call_judge(question: str, reference: str, prediction: str) -> Dict[str, Any]:
    """
    Appelle Mixtral via OpenRouter pour évaluer la qualité de la réponse.

    Sortie:
        {
            "score": float entre 0 et 1,
            "justification": str
        }
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY manquant : score neutre 0.0.")
        return {
            "score": 0.0,
            "justification": "Clé API manquante, évaluation non réalisée.",
        }

    system_prompt = (
        "Tu es un examinateur expert en service client télécom. "
        "On te donne une question d'un client, une réponse de référence (idéale) et "
        "une réponse générée par un agent. "
        "Tu dois attribuer un score entre 0 et 1 (float) qui mesure la similarité "
        "sémantique et la pertinence de la réponse générée par rapport à la référence. "
        "0 signifie totalement incorrect ou hors sujet, 1 signifie parfaitement correct "
        "et aligné avec la réponse de référence. "
        "Explique brièvement ta décision."
    )

    user_prompt = f"""
Question du client:
{question}

Réponse de référence (idéale):
{reference}

Réponse générée par l'agent:
{prediction}

Consignes:
- Analyse le fond (information, exactitude, complétude) plus que la forme.
- Ignore les différences mineures de formulation.
- Renvoie un JSON **strictement valide** de la forme:
{{
  "score": <float entre 0 et 1>,
  "justification": "<texte court en français>"
}}
Ne renvoie rien d'autre que ce JSON.
""".strip()

    body = _build_openrouter_payload(system_prompt, user_prompt)

    try:
        data = _post_openrouter(body, timeout=120)
        content = data["choices"][0]["message"]["content"]
        parsed = _safe_parse_json_content(content)

        if not parsed:
            logger.error("JSON d'évaluation non parsable: %s", content)
            return {
                "score": 0.0,
                "justification": "Impossible de parser la réponse du juge.",
            }

        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        justification = str(parsed.get("justification", "")).strip()
        return {"score": score, "justification": justification}
    except Exception as e:
        logger.error("Erreur lors de l'évaluation LLM-as-a-judge: %s", e)
        return {
            "score": 0.0,
            "justification": "Erreur technique pendant l'évaluation LLM-as-a-judge.",
        }


def main() -> None:
    if not os.path.exists(EVAL_PATH):
        raise FileNotFoundError(f"Fichier d'évaluation introuvable: {EVAL_PATH}")

    df = pd.read_excel(EVAL_PATH)

    expected_columns = {"Question", "Réponse Attendue"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {EVAL_PATH}: {missing}. "
            "Adapte evaluate.py ou le fichier Excel."
        )

    results: List[Dict[str, Any]] = []

    iterator = df.iterrows()
    if tqdm is not None:
        iterator = tqdm(
            df.iterrows(),
            total=len(df),
            desc="Évaluation des questions",
        )

    for idx, row in iterator:
        question = str(row["Question"])
        reference = str(row["Réponse Attendue"])

        logger.info("Question %d/%d", idx + 1, len(df))

        # 1) Appel de l'agent
        try:
            prediction = agent_answer(question)
        except Exception as e:
            logger.error("Erreur lors de l'appel de l'agent: %s", e)
            prediction = "Erreur de l'agent lors de la génération de la réponse."

        # 2) Évaluation par LLM-as-a-judge
        judge_result = call_judge(question, reference, prediction)

        result_row = {
            "index": idx,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "score": judge_result.get("score", 0.0),
            "justification": judge_result.get("justification", ""),
        }
        results.append(result_row)

    # 3) Sauvegarde des résultats
    results_df = pd.DataFrame(results)
    avg_score = results_df["score"].mean() if not results_df.empty else 0.0
    logger.info("Score moyen sur le jeu d'évaluation: %.3f", avg_score)

    results_df.to_excel(OUTPUT_PATH, index=False)
    logger.info("Résultats sauvegardés dans: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
