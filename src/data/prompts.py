COUNTERFACT_PROMPT = (
    "Finish the following statement with the correct fact.\n"
    "Statement: {prompt_stem}"
)

# Standard Alpaca-style prompt for Base Models
SQL_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Write a SQL query to answer the question.\n\n"
    "### Input:\n"
    "{sql_prompt}\n\n"
    "### Context:\n"
    "{sql_context}\n\n"
    "### Response:\n"
)

# Fallback if no context is provided
SQL_PROMPT_NO_CONTEXT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Write a SQL query to answer the question.\n\n"
    "### Input:\n"
    "{sql_prompt}\n\n"
    "### Response:\n"
)

MEDMCQA_PROMPT_TEMPLATE = (
    "Answer the medical question below by choosing the correct option letter.\n"
    "Question: {question}\n"
    "Options:\n"
    "A) {opa}\n"
    "B) {opb}\n"
    "C) {opc}\n"
    "D) {opd}\n"
    "Answer:"
)
