import re


def clean_reply(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<function=.*?</function>", "", text)
    text = text.replace("\n", " ")
    return text.strip()
