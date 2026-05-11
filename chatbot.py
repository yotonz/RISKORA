from utils.helpers import credit_score_band, dti_band, fmt_currency, fmt_percent


def chatbot_response(data: dict, question: str) -> str:
    q = question.lower()

    risk = data.get("risk", "")
    cs = data.get("credit_score", 0)
    dti = data.get("dti", 0.0)
    score = data.get("score", 0)
    name = data.get("applicant_name", "the applicant")
    purpose = data.get("purpose", "")
    reasons = data.get("reasons", [])
    income = data.get("income", 0)
    loan = data.get("loan", 0)
    emi = data.get("emi", 0)

    cs_band = credit_score_band(cs)
    dti_band_label = dti_band(dti)

    if any(k in q for k in ("why", "reason", "cause", "explain", "how come")):
        factors = []
        if cs < 600:
            factors.append(f"Credit score of {cs} ({cs_band}) is below the safe threshold of 600")
        if dti > 45:
            factors.append(f"DTI ratio of {fmt_percent(dti)} ({dti_band_label}) indicates high debt burden")
        if income and loan and loan / income > 4:
            factors.append(f"Loan amount is {loan/income:.1f}× annual income — banks prefer under 4×")
        if reasons:
            factors.extend(reasons)
        if not factors:
            factors.append("All primary metrics are within acceptable ranges")
        factor_list = "\n".join(f"  • {f}" for f in factors)
        return (
            f"**Risk assessment for {name}: {risk}**\n\n"
            f"Key factors:\n{factor_list}\n\n"
            f"Financial score: {score}/100 ({_score_label(score)})"
        )

    if any(k in q for k in ("improve", "better", "fix", "increase", "boost", "enhance")):
        tips = []
        if cs < 750:
            tips.append(f"Raise credit score from {cs} to 750+ by paying bills on time and reducing existing debt")
        if dti > 35:
            monthly_income = income / 12 if income else 0
            ideal_emi = monthly_income * 0.30
            tips.append(
                f"Reduce monthly EMI to under {fmt_currency(ideal_emi)} "
                f"(30% of monthly income) — current DTI is {fmt_percent(dti)}"
            )
        if income and loan and loan / income > 4:
            safe_loan = income * 4
            tips.append(f"Consider reducing loan to {fmt_currency(safe_loan)} (4× income) or less")
        if not tips:
            tips.append("Your financial profile is already strong — maintain on-time payments to keep it that way")
        tip_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tips))
        return f"**Improvement recommendations for {name}:**\n\n{tip_list}"

    if any(k in q for k in ("afford", "can i", "eligible", "qualify", "approval")):
        monthly_income = income / 12 if income else 0
        verdict = "within affordable limits" if dti < 40 else "stretching your budget"
        return (
            f"**Affordability check for {name}:**\n\n"
            f"  • Monthly income: {fmt_currency(monthly_income)}\n"
            f"  • Monthly EMI: {fmt_currency(emi)}\n"
            f"  • DTI ratio: {fmt_percent(dti)} ({dti_band_label}) — {verdict}\n"
            f"  • Ideal DTI: below 35%\n\n"
            f"{'The loan appears manageable.' if dti < 35 else 'Consider a lower EMI or loan amount.'}"
        )

    if any(k in q for k in ("score", "rating", "grade", "financial")):
        return (
            f"**Financial score breakdown for {name}:**\n\n"
            f"  • Overall score: {score}/100 ({_score_label(score)})\n"
            f"  • Credit score: {cs} ({cs_band})\n"
            f"  • DTI ratio: {fmt_percent(dti)} ({dti_band_label})\n\n"
            f"{'Scores above 65 are considered strong.' if score >= 65 else 'Scores below 45 indicate elevated risk.'}"
        )

    if any(k in q for k in ("credit", "cibil", "bureau")):
        advice = {
            "Excellent": "Excellent credit — you qualify for the best interest rates.",
            "Good": "Good credit — eligible for most loan products.",
            "Fair": "Fair credit — approval likely but at higher interest rates.",
            "Below Average": "Below average — lenders may require collateral or a co-applicant.",
            "Poor": "Poor credit — most banks will decline without significant improvement.",
        }
        return (
            f"**Credit score analysis:**\n\n"
            f"  • Score: {cs} — {cs_band}\n"
            f"  • {advice.get(cs_band, '')}\n\n"
            f"Target 750+ for the best loan terms. "
            f"{'You are already there!' if cs >= 750 else f'You need {750 - cs} more points.'}"
        )

    if any(k in q for k in ("dti", "debt", "ratio", "income")):
        return (
            f"**Debt-to-Income analysis:**\n\n"
            f"  • Annual income: {fmt_currency(income)}\n"
            f"  • Annual EMI outflow: {fmt_currency(emi * 12)}\n"
            f"  • DTI ratio: {fmt_percent(dti)} ({dti_band_label})\n\n"
            f"Ideal DTI is below 35%. "
            f"{'You are within a healthy range.' if dti < 35 else 'Reducing EMI commitments would improve your profile.'}"
        )

    if any(k in q for k in ("purpose", "loan type", "category")):
        return (
            f"**Loan purpose: {purpose or 'Not specified'}**\n\n"
            f"Loan purpose does not directly change the risk score, "
            f"but lenders may view {purpose.lower() or 'certain'} loans differently. "
            f"Home and education loans typically attract more favorable terms."
        )

    if any(k in q for k in ("risk", "result", "outcome", "decision", "status")):
        color_word = "green" if risk == "LOW RISK" else "red"
        return (
            f"**Final risk decision: {risk}**\n\n"
            f"The system marked this application as **{risk}** ({color_word} flag).\n"
            f"Financial score: {score}/100 | Credit: {cs} | DTI: {fmt_percent(dti)}\n\n"
            f"{'This application is likely to be approved.' if risk == 'LOW RISK' else 'This application faces significant hurdles for approval.'}"
        )

    return (
        "I can answer questions about:\n"
        "  • **Why** is this result HIGH or LOW risk?\n"
        "  • How to **improve** the profile?\n"
        "  • Can the applicant **afford** this loan?\n"
        "  • What is the financial **score**?\n"
        "  • What does the **credit** score mean?\n"
        "  • What is the **DTI** ratio?\n"
        "  • What is the **risk** decision?\n\n"
        "Try asking one of those!"
    )


def _score_label(score: int) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Poor"
