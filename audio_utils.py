import re

def remove_repeated_words(text):
    pattern = re.compile(r'\b(\S+)(?:\s+\1\b)+', flags=re.IGNORECASE)
    return pattern.sub(r'\1', text)

def remove_overlap(prev_text, new_text):
    max_overlap = min(len(prev_text), len(new_text))
    for i in range(max_overlap, 0, -1):
        if prev_text[-i:] == new_text[:i]:
            return new_text[i:]
    return new_text