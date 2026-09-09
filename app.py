"""
Contrôle de Facturation Transport & Sous-Traitance
---------------------------------------------------
Application Streamlit permettant de comparer un ordre de transport (OT)
initial avec la facture reçue du transporteur, à l'aide de GPT-4o pour
l'extraction et la comparaison des montants.

Lancement :
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import re

import pandas as pd
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# --------------------------------------------------------------------------
# Configuration générale de la page
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Contrôle de Facturation Transport & Sous-Traitance",
    page_icon="🚛",
    layout="wide",
)

st.title("🚛 Contrôle de Facturation Transport & Sous-Traitance")
st.caption(
    "Comparez automatiquement un ordre de transport et la facture reçue "
    "pour détecter les écarts de tarification, de surcharge gazole et de frais annexes."
)

st.divider()

# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def extraire_texte_pdf(fichier) -> str:
    """Extrait le texte brut d'un PDF uploadé via st.file_uploader."""
    lecteur = PdfReader(fichier)
    pages_texte = []
    for page in lecteur.pages:
        contenu = page.extract_text() or ""
        pages_texte.append(contenu)
    return "\n".join(pages_texte)


def nettoyer_json(texte: str) -> dict:
    """Sécurise le parsing JSON renvoyé par le modèle (retire d'éventuels ```json)."""
    texte_nettoye = re.sub(r"^```json|```$", "", texte.strip(), flags=re.MULTILINE).strip()
    return json.loads(texte_nettoye)


def to_float(valeur) -> float:
    """Convertit une valeur potentiellement textuelle en float, sinon 0.0."""
    if valeur is None:
        return 0.0
    if isinstance(valeur, (int, float)):
        return float(valeur)
    try:
        nettoye = str(valeur).replace("€", "").replace("%", "").replace(",", ".").strip()
        return float(nettoye) if nettoye else 0.0
    except ValueError:
        return 0.0


PROMPT_SYSTEME = """Tu es un expert en contrôle de facturation transport routier.
On te fournit le texte extrait d'un ORDRE DE TRANSPORT (le montant convenu à l'origine)
et le texte extrait d'une FACTURE reçue du transporteur (le montant réellement facturé).

Ta mission : extraire et comparer précisément trois postes :
1. Le prix de base HT (prix de transport hors taxes, hors surcharges).
2. Le taux ou montant de surcharge gazole (souvent exprimé en % ou en €).
3. Les frais d'attente / frais annexes (attente, manutention, ADR, palettes, etc.), en cumulant
   les différentes lignes annexes si plusieurs sont présentes.

Pour chaque poste, indique la valeur trouvée dans l'ordre de transport ("convenu") et la valeur
trouvée dans la facture ("facture"). Les surcharges en % doivent être ramenées en montant € si
possible en te basant sur le prix de base ; si ce n'est pas possible, indique le % dans le champ
"unite" et laisse un commentaire dans "observations".

Si une information est introuvable, indique 0 et précise-le dans "observations".

Réponds UNIQUEMENT avec un objet JSON strictement au format suivant, sans texte autour :

{
  "prix_base_ht": {"convenu": <float>, "facture": <float>},
  "surcharge_gazole": {"convenu": <float>, "facture": <float>, "unite": "€ ou %"},
  "frais_annexes": {"convenu": <float>, "facture": <float>},
  "observations": "<synthèse courte des anomalies ou informations manquantes>"
}
"""


def analyser_ecarts(cle_api: str, texte_ot: str, texte_facture: str) -> dict:
    """Envoie les deux textes à GPT-4o et renvoie le JSON structuré des écarts."""
    client = OpenAI(api_key=cle_api)

    message_utilisateur = (
        "=== TEXTE DE L'ORDRE DE TRANSPORT (montant convenu) ===\n"
        f"{texte_ot[:15000]}\n\n"
        "=== TEXTE DE LA FACTURE REÇUE (montant facturé) ===\n"
        f"{texte_facture[:15000]}"
    )

    reponse = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT_SYSTEME},
            {"role": "user", "content": message_utilisateur},
        ],
    )
    contenu = reponse.choices[0].message.content
    return nettoyer_json(contenu)


def generer_email_litige(cle_api: str, tableau: pd.DataFrame, observations: str) -> str:
    """Génère un brouillon de mail de litige à partir du tableau d'écarts."""
    client = OpenAI(api_key=cle_api)

    recap = tableau.to_string(index=False)

    prompt = f"""Rédige un e-mail professionnel et courtois de litige de facturation transport,
à envoyer au transporteur, à partir du récapitulatif d'écarts ci-dessous.

L'e-mail doit :
- rappeler l'objet (contrôle de facturation sur le transport concerné),
- lister clairement les écarts constatés poste par poste avec les montants convenus vs facturés,
- demander l'émission d'un avoir ou d'une facture rectificative pour le montant total de l'écart,
- rester factuel, ferme mais courtois,
- se terminer par une formule de politesse standard.

Récapitulatif des écarts :
{recap}

Observations complémentaires : {observations}

Ne mets ni objet d'e-mail générique creux ni placeholders du type [Nom] excessifs ;
utilise "Madame, Monsieur," en ouverture et signe "Le service Achats / Approvisionnement".
"""

    reponse = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return reponse.choices[0].message.content


def style_ecart(valeur: float) -> str:
    """Colore en rouge les écarts positifs (facturé > convenu), en vert les négatifs."""
    if valeur > 0:
        return "color: #d62728; font-weight: bold;"
    if valeur < 0:
        return "color: #2ca02c; font-weight: bold;"
    return ""


# --------------------------------------------------------------------------
# Barre latérale : clé API
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Configuration")
    cle_api_openai = st.text_input(
        "Clé API OpenAI",
        type="password",
        placeholder="sk-...",
        help="Votre clé n'est jamais stockée ni envoyée ailleurs qu'à l'API OpenAI.",
    )
    st.markdown("---")
    st.caption(
        "Modèle utilisé : `gpt-4o`. Le texte des PDF est extrait localement "
        "puis transmis au modèle pour l'analyse comparative."
    )

# --------------------------------------------------------------------------
# Zone de dépôt des deux PDF
# --------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Ordre de transport")
    fichier_ot = st.file_uploader(
        "Déposer l'ordre de transport initial",
        type=["pdf"],
        key="upload_ot",
    )

with col2:
    st.subheader("🧾 Facture reçue")
    fichier_facture = st.file_uploader(
        "Déposer la facture reçue",
        type=["pdf"],
        key="upload_facture",
    )

st.divider()

# --------------------------------------------------------------------------
# État persistant de la session
# --------------------------------------------------------------------------
if "resultat_analyse" not in st.session_state:
    st.session_state.resultat_analyse = None
if "tableau_ecarts" not in st.session_state:
    st.session_state.tableau_ecarts = None
if "email_litige" not in st.session_state:
    st.session_state.email_litige = None

# --------------------------------------------------------------------------
# Lancement de l'analyse
# --------------------------------------------------------------------------
bouton_analyse = st.button("🔍 Lancer l'analyse des écarts", type="primary", use_container_width=True)

if bouton_analyse:
    if not cle_api_openai:
        st.error("Merci de renseigner votre clé API OpenAI dans la barre latérale.")
    elif not fichier_ot or not fichier_facture:
        st.error("Merci de déposer à la fois l'ordre de transport et la facture reçue.")
    else:
        with st.spinner("Extraction du texte des PDF..."):
            texte_ot = extraire_texte_pdf(fichier_ot)
            texte_facture = extraire_texte_pdf(fichier_facture)

        if not texte_ot.strip() or not texte_facture.strip():
            st.warning(
                "Le texte extrait d'un des PDF est vide (document scanné en image ?). "
                "L'analyse risque d'être incomplète."
            )

        try:
            with st.spinner("Analyse comparative en cours via GPT-4o..."):
                resultat = analyser_ecarts(cle_api_openai, texte_ot, texte_facture)
            st.session_state.resultat_analyse = resultat
            st.session_state.email_litige = None  # réinitialise l'email précédent
            st.success("Analyse terminée.")
        except Exception as erreur:
            st.error(f"Erreur lors de l'appel à l'API OpenAI : {erreur}")
            st.session_state.resultat_analyse = None

# --------------------------------------------------------------------------
# Affichage du tableau récapitulatif
# --------------------------------------------------------------------------
if st.session_state.resultat_analyse:
    resultat = st.session_state.resultat_analyse

    lignes = []
    libelles = {
        "prix_base_ht": "Prix de base HT",
        "surcharge_gazole": "Surcharge gazole",
        "frais_annexes": "Frais d'attente / annexes",
    }
    for cle, libelle in libelles.items():
        poste = resultat.get(cle, {})
        convenu = to_float(poste.get("convenu"))
        facture = to_float(poste.get("facture"))
        lignes.append(
            {
                "Poste": libelle,
                "Montant convenu (€)": round(convenu, 2),
                "Montant facturé (€)": round(facture, 2),
                "Écart (€)": round(facture - convenu, 2),
            }
        )

    df_ecarts = pd.DataFrame(lignes)
    ligne_totale = pd.DataFrame(
        [
            {
                "Poste": "TOTAL",
                "Montant convenu (€)": round(df_ecarts["Montant convenu (€)"].sum(), 2),
                "Montant facturé (€)": round(df_ecarts["Montant facturé (€)"].sum(), 2),
                "Écart (€)": round(df_ecarts["Écart (€)"].sum(), 2),
            }
        ]
    )
    df_final = pd.concat([df_ecarts, ligne_totale], ignore_index=True)
    st.session_state.tableau_ecarts = df_final

    st.subheader("📊 Récapitulatif des écarts")

    df_style = df_final.style.applymap(style_ecart, subset=["Écart (€)"])
    df_style = df_style.format(
        {"Montant convenu (€)": "{:.2f}", "Montant facturé (€)": "{:.2f}", "Écart (€)": "{:+.2f}"}
    )
    st.dataframe(df_style, use_container_width=True, hide_index=True)

    ecart_total = df_final.loc[df_final["Poste"] == "TOTAL", "Écart (€)"].iloc[0]
    if ecart_total > 0:
        st.error(f"⚠️ Écart total en faveur du transporteur : **+{ecart_total:.2f} €** (surfacturation)")
    elif ecart_total < 0:
        st.info(f"ℹ️ Écart total en votre faveur : **{ecart_total:.2f} €** (sous-facturation)")
    else:
        st.success("✅ Aucun écart détecté entre l'ordre de transport et la facture.")

    if resultat.get("observations"):
        st.caption(f"💬 Observations du modèle : {resultat['observations']}")

    st.divider()

    # ----------------------------------------------------------------
    # Génération du brouillon de mail de litige
    # ----------------------------------------------------------------
    st.subheader("✉️ Brouillon de mail de litige")

    if st.button("✍️ Générer le brouillon de mail de litige", use_container_width=True):
        if not cle_api_openai:
            st.error("Merci de renseigner votre clé API OpenAI dans la barre latérale.")
        else:
            try:
                with st.spinner("Rédaction du mail en cours..."):
                    email_genere = generer_email_litige(
                        cle_api_openai,
                        st.session_state.tableau_ecarts,
                        resultat.get("observations", ""),
                    )
                st.session_state.email_litige = email_genere
            except Exception as erreur:
                st.error(f"Erreur lors de la génération du mail : {erreur}")

    if st.session_state.email_litige:
        st.caption("Cliquez sur l'icône en haut à droite du bloc ci-dessous pour copier le mail.")
        st.code(st.session_state.email_litige, language="text")
