def get_price_query(model: str, input_token: int, output_token: int) -> float:
    """
    Calcule le prix d'une requête LLM en fonction du modèle et du nombre de tokens.
    """
    dict_price = {
        "ministral-8b-latest": {"input": 0.10, "output": 0.10},
        "ministral-3b-latest": {"input": 0.04, "output": 0.04},
        "codestral-latest": {"input": 0.20, "output": 0.60},
        "mistral-large-latest": {"input": 2, "output": 6},
    }
    price = dict_price.get(model, {"input": 0, "output": 0})
    return ((price["input"] / 10**6) * input_token) + ((price["output"] / 10**6) * output_token)


def get_energy_usage(response):
    """
    Récupère la consommation d'énergie et l'empreinte carbone depuis la réponse LLM.
    """
    try:
        # 🔹 Vérifier si 'usage' est présent dans la réponse du modèle
        if "usage" in response:
            completion_tokens = response["usage"].get("completion_tokens", 0)
            prompt_tokens = response["usage"].get("prompt_tokens", 0)

            # ✅ Simulation de l'énergie utilisée (données non fournies directement par Mistral)
            # ⚠️ Ces valeurs sont fictives (besoin d'un modèle de conversion réel)
            energy_usage = (completion_tokens + prompt_tokens) * 0.0001  # Exemple : 0.1 Wh par token
            gwp = energy_usage * 0.05  # Exemple : 0.05 gCO2e par Wh

            return energy_usage, gwp

    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération de l'impact énergétique : {e}")

    return None, None  # Retourne None si impossible d'extraire les données
