import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _appeler_gemini(image_bytes: bytes, mime_type: str, prompt: str,
                    max_tokens: int = 1024, temperature: float = 0.1) -> str:
    """Appel générique à l'API Gemini avec une image et un prompt."""
    contenu = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        types.Part.from_text(text=prompt),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contenu,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text.strip()


def est_architecture_cloud(image_bytes: bytes,
                            mime_type: str = "image/png") -> tuple[bool, str]:
    """
    Vérifie si l'image représente une architecture cloud.
    Returns:
        (True, "")        si c'est une architecture cloud
        (False, "raison") sinon
    """
    prompt = """Regarde attentivement cette image et réponds UNIQUEMENT par un JSON avec ce format exact :
{
  "est_architecture_cloud": true ou false,
  "raison": "explication courte en français"
}
Une architecture cloud inclut : diagrammes AWS/Azure/GCP, schémas réseau, infrastructure as code,
topologies de services cloud, diagrammes de déploiement, schémas de microservices, etc.
Ce n'est PAS une architecture cloud si c'est : une photo, un texte, un logo, une interface utilisateur
sans composants cloud, un document, une capture d'écran d'application, etc."""

    try:
        resultat = _appeler_gemini(
            image_bytes, mime_type, prompt,
            max_tokens=256,
            temperature=0.0,
        )
        resultat = resultat.replace("```json", "").replace("```", "").strip()
        data = json.loads(resultat)
        return data.get("est_architecture_cloud", False), data.get("raison", "")
    except Exception:
        # En cas d'erreur de parsing, on laisse passer
        return True, ""


def analyser_image_architecture(
    image_bytes: bytes,
    mime_type: str = "image/png",
    prompt_path: str = "prompts/gemini_prompt.txt",
) -> str:
    """
    Analyse une image d'architecture cloud et retourne une description
    structurée exhaustive destinée à la génération Terraform.
    Lève une ValueError si l'image n'est pas une architecture cloud.
    """
    # Étape 1 : validation de l'image
    valide, raison = est_architecture_cloud(image_bytes, mime_type)
    if not valide:
        raise ValueError(f"IMAGE_NON_VALIDE:{raison}")

    # Étape 2 : chargement du prompt depuis le fichier
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()          # ← bug corrigé (f.read() et non [f.read])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Fichier prompt introuvable : {prompt_path}\n"
            "Crée le fichier prompts/gemini_prompt.txt dans ton projet."
        )

    # Étape 3 : analyse complète de l'image
    contenu = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        types.Part.from_text(text=prompt_text),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contenu,
        config=types.GenerateContentConfig(
            max_output_tokens=8192,   # ← augmenté : analyse exhaustive
            temperature=0.1,          # ← faible pour rester factuel
        ),
    )
    return response.text.strip()