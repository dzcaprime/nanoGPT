SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
EOT = "<|endoftext|>"

VALID_ROLES = {SYSTEM, USER, ASSISTANT}


def _validate_messages(messages):
    if not messages:
        raise ValueError("messages must not be empty")
    last_role = None
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role: {role}")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        if role == SYSTEM and last_role is not None:
            raise ValueError("system message must be first")
        if role == ASSISTANT and last_role != USER:
            raise ValueError("assistant message must follow user")
        if role == USER and last_role == USER:
            raise ValueError("user message cannot follow user")
        last_role = role


def format_messages(messages, include_eot=True):
    text, _ = format_messages_with_spans(messages, include_eot=include_eot)
    return text


def format_messages_with_spans(messages, include_eot=True):
    _validate_messages(messages)
    parts = []
    assistant_spans = []
    cursor = 0
    for i, message in enumerate(messages):
        role = message["role"]
        header = f"<|{role}|>\n"
        parts.append(header)
        cursor += len(header)
        content = message["content"]
        start = cursor
        parts.append(content)
        cursor += len(content)
        if role == ASSISTANT:
            assistant_spans.append((start, cursor))
        if i != len(messages) - 1:
            parts.append("\n")
            cursor += 1
    if include_eot:
        parts.append(EOT)
    return "".join(parts), assistant_spans


def build_loss_mask(token_offsets, assistant_spans):
    mask = []
    for start, end in token_offsets:
        trainable = any(start < span_end and end > span_start for span_start, span_end in assistant_spans)
        mask.append(1 if trainable else 0)
    return mask


def encode_with_assistant_labels(enc, messages):
    ids = []
    labels = []
    for i, message in enumerate(messages):
        role = message["role"]
        header_ids = enc.encode_ordinary(f"<|{role}|>\n")
        ids.extend(header_ids)
        labels.extend([-1] * len(header_ids))

        content_ids = enc.encode_ordinary(message["content"])
        ids.extend(content_ids)
        if role == ASSISTANT:
            labels.extend(content_ids)
        else:
            labels.extend([-1] * len(content_ids))

        if i != len(messages) - 1:
            newline_ids = enc.encode_ordinary("\n")
            ids.extend(newline_ids)
            labels.extend([-1] * len(newline_ids))

    eot_ids = enc.encode(EOT, allowed_special={EOT})
    ids.extend(eot_ids)
    labels.extend(eot_ids if messages[-1]["role"] == ASSISTANT else [-1] * len(eot_ids))
    return ids, labels
