"""Validate the full supplied option space without changing model probabilities."""
import math


def validate_response(request, response):
    questions = request["questions"]
    answers = response["answers"]
    if set(answers) != set(questions):
        raise ValueError("Missing or extra decisions")
    for key, question in questions.items():
        answer = answers[key]
        if answer["type"] != question["type"]:
            raise ValueError("Wrong decision type")
        if question["type"] == "noul":
            value = answer["noul"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid noul probability")
            continue
        options = (set(question["criteria"]) if question["type"] == "choice"
                   else {str(i) for i in range(len(question["criteria"]))})
        probs = answer["probabilities"]
        if set(probs) != options or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
            raise ValueError("Invalid or incomplete option probabilities")
        if abs(sum(probs.values()) - 1) > 0.001:
            raise ValueError("Probabilities do not sum to one")
        if question["type"] == "choice" and answer["choice"] != max(probs, key=probs.get):
            raise ValueError("Choice does not match argmax")
    if response["usage"]["output_tokens"] != 0:
        raise ValueError("Unexpected generated tokens")
    tokens = response["usage"]["input_tokens"]
    if type(tokens) is not int or tokens < 1:
        raise ValueError("Invalid input token count")
