import warnings
# Suppress specific RoBERTa/Hugging Face warnings early
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized:.*", category=UserWarning)
warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.", category=UserWarning)

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_cosine_similarity(current, previous):
    """Calculate cosine similarity between current text and previous texts."""
    if not previous:
        return 0.0
    current_embedding = _embedding_model.encode([current])
    previous_embeddings = _embedding_model.encode(previous)
    similarities = cosine_similarity(current_embedding, previous_embeddings)[0]
    return float(np.mean(similarities))


def get_self_bleu(candidate, others):
    """Calculate BLEU score between current text and previous texts."""
    if not others:
        return 0.0
    smooth = SmoothingFunction().method1
    scores = [sentence_bleu([o.split()], candidate.split(), smoothing_function=smooth) for o in others]
    return float(np.mean(scores))


def get_bertscore(candidate, others, lang="en"):
    """Calculate BERTScore between current text and previous texts."""
    if not others:
        return 0.0
    cands = [candidate] * len(others)
    P, R, F1 = bert_score_fn(cands, others, lang=lang, verbose=False)
    return float(torch.mean(F1).item())


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