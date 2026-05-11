def fmt_currency(amount: float) -> str:
    """Format a number as Indian-style currency string (e.g. ₹12,50,000)."""
    amount = int(amount)
    s = str(abs(amount))
    if len(s) <= 3:
        result = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        result = ",".join(reversed(groups)) + "," + last3
    return f"{'−' if amount < 0 else ''}₹{result}"


def fmt_percent(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def risk_color(risk: str) -> str:
    mapping = {
        "HIGH RISK": "#ef4444",
        "LOW RISK": "#22c55e",
        "CAUTION": "#f97316",
    }
    return mapping.get(risk, "#94a3b8")


def credit_score_band(score: int) -> str:
    if score >= 750:
        return "Excellent"
    if score >= 700:
        return "Good"
    if score >= 650:
        return "Fair"
    if score >= 600:
        return "Below Average"
    return "Poor"


def dti_band(dti: float) -> str:
    if dti < 20:
        return "Healthy"
    if dti < 35:
        return "Manageable"
    if dti < 50:
        return "Elevated"
    if dti < 65:
        return "High"
    return "Very High"
