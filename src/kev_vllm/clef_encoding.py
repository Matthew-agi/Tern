"""Application schema spans transported to the worker alongside token IDs."""


def schema_metadata(record):
    return {"record_id": record.record_id, "questions": [
        {"question_id": q.question_id, "question_type": q.question_type,
         "question_span": list(q.question_span), "option_spans": list(map(list, q.option_spans)),
         "option_ids": list(q.option_ids)} for q in record.questions]}
