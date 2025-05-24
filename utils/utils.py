import numpy as np
import tiktoken


def read_npy_file(file_path):

    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def format_ocr_tokens_to_string(ocr_tokens):
    text = ""
    for page in ocr_tokens:
        text += "<page> "
        for token in page:
            text += token + " "
        text += "</page> "
    return text.strip()


def estimate_text_length(ocr_tokens):
    return len(ocr_tokens)


def estimate_words_count(text):
    text_split = text.split()
    text_alpha = [word for word in text_split if word.isalpha()]
    return len(text_alpha)


def estimate_llm_tokens_count(text, encoding="o200k_base"):
    encoding = tiktoken.get_encoding(encoding)
    tokens = encoding.encode(text)
    return len(tokens)
