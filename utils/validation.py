import re


def validate(age, income, loan, credit_score, emi):
    if age < 18 or age > 65:
        return "Age must be between 18 and 65"

    if income <= 0:
        return "Annual income must be greater than 0"

    if loan <= 0:
        return "Loan amount must be greater than 0"

    if credit_score < 300 or credit_score > 900:
        return "Credit score must be between 300 and 900"

    if emi <= 0:
        return "Monthly EMI must be greater than 0"

    if emi * 12 > income:
        return "Annual EMI (EMI × 12) cannot exceed annual income"

    if loan > income * 8:
        return "Loan amount is too high relative to income (max 8× annual income)"

    return "OK"


def validate_email(email: str) -> bool:
    if not email:
        return True
    return bool(re.match(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$", email))


def validate_phone(phone: str) -> bool:
    if not phone:
        return True
    digits = re.sub(r"[\s\-+()]", "", phone)
    return digits.isdigit() and 7 <= len(digits) <= 15
