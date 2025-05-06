import warnings
# Suppress ALL UserWarnings from transformers library - might hide other potentially useful warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
# Keep the specific bert_score filter just in case
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized:.*", category=UserWarning)

# Disable all logging messages from transformers
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load models once at module level
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
_bert_scorer = None  # Will be lazily initialized on first use

def _get_bert_scorer():
    """Lazily initialize the BERTScorer to avoid loading it if it's not used."""
    global _bert_scorer
    if _bert_scorer is None:
        # Temporarily disable all output
        original_stderr = None
        if not logging.getLogger().isEnabledFor(logging.INFO):
            import sys
            original_stderr = sys.stderr
            sys.stderr = open('/dev/null', 'w')
        
        try:
            print("[INFO] Initializing BERTScore model (one-time operation)...")
            _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            print("[INFO] BERTScore model initialized.")
        finally:
            # Restore stderr if we changed it
            if original_stderr:
                sys.stderr.close()
                sys.stderr = original_stderr
    
    return _bert_scorer

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
    
    # Get the pre-loaded BERTScorer
    scorer = _get_bert_scorer()
    
    # Calculate scores using cached model
    cands = [candidate] * len(others)
    P, R, F1 = scorer.score(cands, others)
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