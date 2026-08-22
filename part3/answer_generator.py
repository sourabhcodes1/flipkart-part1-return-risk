import re

from rag_query import search_policy
from return_risk_tool import check_return_risk


def _extract_number(question, patterns, field_name, required=True):
    """
    Try several regex patterns and return the first matching number.
    """
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    if required:
        raise ValueError(f"Could not find {field_name} in the question.")

    return None


def _extract_return_risk_values(question):
    """
    Extract all inputs required by check_return_risk().
    """

    q = question.lower()

    # ---------------------------------------------------------
    # Price
    # ---------------------------------------------------------
    price = _extract_number(
        q,
        [
            r"priced\s+at\s*(\d+(?:\.\d+)?)",
            r"price\s*(?:is|of)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)",
            r"(?:₹|rs\.?|inr)\s*(\d+(?:\.\d+)?)",
        ],
        "price",
    )

    # ---------------------------------------------------------
    # Discount
    # ---------------------------------------------------------
    discount = _extract_number(
        q,
        [
            r"(\d+(?:\.\d+)?)\s*%\s*discount",
            r"discount\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*%",
        ],
        "discount",
    )

    # ---------------------------------------------------------
    # Customer tenure
    # ---------------------------------------------------------
    tenure = _extract_number(
        q,
        [
            r"(\d+)\s*days?\s*customer\s*tenure",
            r"customer\s*tenure\s*(?:of|is)?\s*(\d+)\s*days?",
        ],
        "customer tenure",
    )

    # ---------------------------------------------------------
    # Previous orders
    # Handles:
    # 1 previous order
    # 2 previous orders
    # ---------------------------------------------------------
    previous_orders = _extract_number(
        q,
        [
            r"(\d+)\s*previous\s*orders?",
            r"previous\s*orders?\s*(?:of|is)?\s*(\d+)",
        ],
        "previous orders",
    )

    # ---------------------------------------------------------
    # Previous returns
    # Handles:
    # 0 previous return
    # 2 previous returns
    # ---------------------------------------------------------
    previous_returns = _extract_number(
        q,
        [
            r"(\d+)\s*previous\s*returns?",
            r"previous\s*returns?\s*(?:of|is)?\s*(\d+)",
        ],
        "previous returns",
    )

    # ---------------------------------------------------------
    # Delivery distance
    # ---------------------------------------------------------
    distance = _extract_number(
        q,
        [
            r"(\d+(?:\.\d+)?)\s*km\s*delivery\s*distance",
            r"delivery\s*distance\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*km",
        ],
        "delivery distance",
    )

    # ---------------------------------------------------------
    # Delivery days
    # ---------------------------------------------------------
    delivery_days = _extract_number(
        q,
        [
            r"(\d+)\s*delivery\s*days?",
            r"delivery\s*days?\s*(?:of|is)?\s*(\d+)",
        ],
        "delivery days",
    )

    # ---------------------------------------------------------
    # Rating
    # ---------------------------------------------------------
    rating = _extract_number(
        q,
        [
            r"rating\s*(?:is|of)?\s*(\d+(?:\.\d+)?)",
            r"rating\s*(\d+(?:\.\d+)?)",
        ],
        "rating",
    )

    # ---------------------------------------------------------
    # Product category
    # ---------------------------------------------------------
    if "apparel" in q or "clothing" in q or "fashion" in q:
        product_category = "Apparel"
    elif "electronics" in q or "electronic" in q:
        product_category = "Electronics"
    elif "grocery" in q or "groceries" in q:
        product_category = "Grocery"
    else:
        product_category = "Apparel"

    # ---------------------------------------------------------
    # Payment method
    # ---------------------------------------------------------
    if (
        "prepaid upi" in q
        or "upi payment" in q
        or "upi" in q
        or "prepaid" in q
    ):
        payment_method = "Prepaid_UPI"
    elif (
        "cod" in q
        or "cash on delivery" in q
        or "cash payment" in q
        or "cash" in q
    ):
        payment_method = "COD"
    else:
        payment_method = "Prepaid_UPI"

    # ---------------------------------------------------------
    # Weekend order
    # ---------------------------------------------------------
    is_weekend_order = (
        "weekend order" in q
        or "weekend" in q
        or "saturday" in q
        or "sunday" in q
    )

    return {
        "product_category": product_category,
        "price_inr": price,
        "discount_pct": discount,
        "payment_method": payment_method,
        "customer_tenure_days": int(tenure),
        "num_previous_orders": int(previous_orders),
        "num_previous_returns": int(previous_returns),
        "delivery_distance_km": distance,
        "delivery_days": int(delivery_days),
        "is_weekend_order": is_weekend_order,
        "rating_given": rating,
    }


def _format_policy_result(result):
    """
    Convert a policy search result into clean text.
    """

    if not result:
        return None

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        text = result.get("text")

        if text:
            return str(text).strip()

        # Fallback if the result uses another common field.
        for key in ("content", "answer", "policy"):
            if result.get(key):
                return str(result[key]).strip()

    return str(result).strip()


def generate_answer(question):
    """
    Generate an answer for:
    1. Return-risk questions
    2. Normal policy questions
    """

    if not isinstance(question, str):
        return "Please enter a valid question."

    question = question.strip()

    if not question:
        return "Please enter a question."

    q = question.lower()

    # =========================================================
    # RETURN-RISK QUESTIONS
    # =========================================================
    if "return risk" in q or (
        "risk" in q
        and (
            "previous returns" in q
            or "previous return" in q
            or "delivery distance" in q
            or "customer tenure" in q
        )
    ):
        try:
            values = _extract_return_risk_values(question)

            result = check_return_risk(
                values["product_category"],
                values["price_inr"],
                values["discount_pct"],
                values["payment_method"],
                values["customer_tenure_days"],
                values["num_previous_orders"],
                values["num_previous_returns"],
                values["delivery_distance_km"],
                values["delivery_days"],
                values["is_weekend_order"],
                values["rating_given"],
            )

            if not isinstance(result, dict):
                return f"Return risk result: {result}"

            probability = result.get("return_risk_probability")
            risk_bucket = result.get("risk_bucket")

            if probability is None:
                probability = result.get("probability")

            if risk_bucket is None:
                risk_bucket = result.get("risk")

            if probability is None or risk_bucket is None:
                return f"Return risk result: {result}"

            return (
                f"Return risk probability: {float(probability):.4f}\n"
                f"Risk bucket: {risk_bucket}"
            )

        except Exception as error:
            return f"Unable to calculate return risk: {error}"

    # =========================================================
    # NORMAL POLICY QUESTIONS
    # =========================================================
    try:
        results = search_policy(question, top_k=3)

        if not results:
            return "I could not find relevant information in the policy knowledge base."

        best_result = results[0]

        answer = _format_policy_result(best_result)

        if not answer:
            return "I could not find relevant information in the policy knowledge base."

        return answer

    except Exception as error:
        return f"Unable to answer the question: {error}"


# =============================================================
# COMMAND-LINE TEST MODE
# =============================================================
if __name__ == "__main__":
    print("AI SUPPORT ASSISTANT")
    print("Type your question.")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() == "exit":
                print("Assistant: Goodbye!")
                break

            if not question:
                continue

            answer = generate_answer(question)
            print(f"\nAssistant: {answer}\n")

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break

        except Exception as error:
            print(f"\nAssistant: {error}\n")