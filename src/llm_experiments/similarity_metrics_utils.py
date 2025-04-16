import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_cosine_similarity(current, previous):
    if not previous:
        return None
    current_embedding = _embedding_model.encode([current])
    previous_embeddings = _embedding_model.encode(previous)
    similarities = cosine_similarity(current_embedding, previous_embeddings)[0]
    return np.mean(similarities)


def get_self_bleu(candidate, others):
    if not others:
        return None
    smooth = SmoothingFunction().method1
    scores = [sentence_bleu([o.split()], candidate.split(), smoothing_function=smooth) for o in others]
    return np.mean(scores)


def get_bertscore(candidate, others, lang="en"):
    if not others:
        return None
    cands = [candidate] * len(others)
    P, R, F1 = bert_score(cands, others, lang=lang, verbose=False)
    return float(np.mean(F1))


def get_llm_similarity_category(llama_fn, prompt, completion, previous):
    if not previous:
        return None, None
    judgment_prompt = f"""
        Compare these AI responses to the prompt: \"{prompt}\"

        Previous responses:
        {chr(10).join([f"- {c}" for c in previous])}

        Current response:
        - {completion}

        How similar is the current response to the previous responses in terms of meaning and content?
        Choose one category:
        - VERY_DIFFERENT
        - SOMEWHAT_DIFFERENT
        - SOMEWHAT_SIMILAR
        - VERY_SIMILAR
        - IDENTICAL

        Respond with only the category name.
    """
    category = llama_fn(judgment_prompt).strip()
    score_map = {
        "VERY_DIFFERENT": 0.0,
        "SOMEWHAT_DIFFERENT": 0.25,
        "SOMEWHAT_SIMILAR": 0.75,
        "VERY_SIMILAR": 0.9,
        "IDENTICAL": 1.0
    }
    return category, score_map.get(category)