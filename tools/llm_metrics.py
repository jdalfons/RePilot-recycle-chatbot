from typing import Tuple, Optional

def get_price_query(model: str, input_token: int, output_token: int) -> float:
    """
    Computes the cost of an LLM query based on the model and the number of tokens used.

    Args:
        model (str): The LLM model name.
        input_token (int): Number of input tokens used.
        output_token (int): Number of output tokens generated.

    Returns:
        float: The total cost of the query.
    """
    pricing = {
        "ministral-8b-latest": {"input": 0.10, "output": 0.10},
        "ministral-3b-latest": {"input": 0.04, "output": 0.04},
        "codestral-latest": {"input": 0.20, "output": 0.60},
        "mistral-large-latest": {"input": 2, "output": 6},
    }
    price = pricing.get(model, {"input": 0, "output": 0})
    return ((price["input"] / 10**6) * input_token) + ((price["output"] / 10**6) * output_token)


def get_energy_usage(response: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts energy consumption and carbon footprint from the LLM response.

    Args:
        response (dict): The response object from the LLM.

    Returns:
        Tuple[Optional[float], Optional[float]]: (energy_usage in Wh, global warming potential in gCO2e).
    """
    try:
        if "usage" in response:
            completion_tokens = response["usage"].get("completion_tokens", 0)
            prompt_tokens = response["usage"].get("prompt_tokens", 0)

            # 🔹 Estimated energy consumption (requires real-world calibration)
            energy_usage = (completion_tokens + prompt_tokens) * 0.0001  # Example: 0.1 Wh per token
            gwp = energy_usage * 0.05  # Example: 0.05 gCO2e per Wh

            return energy_usage, gwp

    except Exception as e:
        print(f"⚠️ Error extracting energy impact: {e}")

    return None, None  # Returns None if data extraction fails
