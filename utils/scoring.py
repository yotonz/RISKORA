def calculate_dti(income: float, emi: float) -> float:
    """Return annualized DTI: (monthly EMI × 12) / annual income × 100."""
    if income <= 0:
        return 0.0
    return round((emi * 12 / income) * 100, 2)


def financial_score(credit_score: int, dti: float) -> int:
    score = 0

    if credit_score >= 750:
        score += 50
    elif credit_score >= 700:
        score += 42
    elif credit_score >= 650:
        score += 34
    elif credit_score >= 600:
        score += 22
    else:
        score += 10

    if dti < 20:
        score += 50
    elif dti < 35:
        score += 40
    elif dti < 50:
        score += 28
    elif dti < 65:
        score += 14
    else:
        score += 5

    return score


def score_label(score: int) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Poor"
