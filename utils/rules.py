def rule_override(age: int, credit_score: int, dti: float, loan: float, income: float):
    reasons = []

    if credit_score < 550:
        reasons.append(f"Very low credit score ({credit_score})")
    elif credit_score < 600:
        reasons.append(f"Low credit score ({credit_score}) — below safe threshold")

    if dti > 60:
        reasons.append(f"Very high debt-to-income ratio ({dti:.1f}%)")
    elif dti > 45:
        reasons.append(f"Elevated debt-to-income ratio ({dti:.1f}%)")

    if income > 0:
        loan_ratio = loan / income
        if loan_ratio > 6:
            reasons.append(f"Loan amount is {loan_ratio:.1f}× annual income (max safe: 6×)")
        elif loan_ratio > 4:
            reasons.append(f"Loan amount is {loan_ratio:.1f}× annual income — moderately high")

    hard_fail = (
        credit_score < 550
        or dti > 60
        or (income > 0 and loan / income > 6)
    )

    if hard_fail:
        return "HIGH", reasons

    if reasons:
        return "CAUTION", reasons

    return "OK", []
