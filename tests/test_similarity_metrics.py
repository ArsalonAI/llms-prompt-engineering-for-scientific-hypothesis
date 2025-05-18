import unittest
import numpy as np
from src.similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore

class TestSimilarityMetrics(unittest.TestCase):

    def test_get_cosine_similarity(self):
        # Test with identical texts
        text1 = "This is a test sentence."
        texts_previous_identical = ["This is a test sentence."]
        similarity_identical = get_cosine_similarity(text1, texts_previous_identical)
        self.assertAlmostEqual(similarity_identical, 1.0, places=2, msg="Identical texts should have cosine similarity near 1.0")

        # Test with very different texts
        text2 = "Another completely different sentence."
        texts_previous_different = ["This is a test sentence."]
        similarity_different = get_cosine_similarity(text2, texts_previous_different)
        self.assertTrue(0.0 <= similarity_different < 0.8, msg="Very different texts should have lower cosine similarity")

        # Test with somewhat similar texts
        text3 = "This is a test phrase."
        texts_previous_somewhat = ["This is a test sentence."]
        similarity_somewhat = get_cosine_similarity(text3, texts_previous_somewhat)
        self.assertTrue(0.5 < similarity_somewhat < 1.0, msg="Somewhat similar texts should have intermediate cosine similarity")
        
        # Test with multiple previous texts
        text4 = "A common phrase."
        texts_previous_multiple = ["A common sentence.", "A similar phrase here.", "This is common wording."]
        similarity_multiple = get_cosine_similarity(text4, texts_previous_multiple)
        self.assertTrue(0.0 <= similarity_multiple <= 1.0, msg="Cosine similarity with multiple texts should be between 0 and 1")

        # Test with empty previous list
        text_empty_prev = "Any sentence."
        texts_previous_empty = []
        similarity_empty = get_cosine_similarity(text_empty_prev, texts_previous_empty)
        self.assertEqual(similarity_empty, 0.0, msg="Cosine similarity with empty previous list should be 0.0")

    def test_get_self_bleu(self):
        candidate1 = "this is a test"
        others1 = ["this is a test"]
        bleu1 = get_self_bleu(candidate1, others1)
        self.assertAlmostEqual(bleu1, 1.0, places=2, msg="Identical candidate and other should have BLEU score near 1.0")

        candidate2 = "this is a completely different sentence"
        others2 = ["this is a test"]
        bleu2 = get_self_bleu(candidate2, others2)
        self.assertTrue(0.0 <= bleu2 < 0.5, msg="Very different sentences should have low BLEU score")

        candidate3 = "this test is a" # Permutation
        others3 = ["this is a test"]
        bleu3 = get_self_bleu(candidate3, others3)
        self.assertTrue(0.2 < bleu3 < 0.8, msg="Permuted sentence should have intermediate BLEU score") # BLEU is sensitive to order

        candidate4 = "the cat sat on the mat"
        others4 = ["the cat was on the mat", "a cat sat on the mat"]
        bleu4 = get_self_bleu(candidate4, others4)
        self.assertTrue(0.0 <= bleu4 <= 1.0, msg="BLEU score with multiple others should be between 0 and 1")
        
        # Test with empty others list
        candidate_empty_others = "Any sentence."
        others_empty = []
        bleu_empty = get_self_bleu(candidate_empty_others, others_empty)
        self.assertEqual(bleu_empty, 0.0, msg="Self BLEU with empty others list should be 0.0")

    def test_get_bertscore(self):
        # Note: BERTScore can be slow and requires model download on first run.
        # These tests are more like integration tests for this function.
        candidate1 = "This is a test sentence."
        others1 = ["This is a test sentence."]
        # BERTScore F1 for identical sentences should be high (close to 1.0)
        bert_f1_1 = get_bertscore(candidate1, others1)
        self.assertAlmostEqual(bert_f1_1, 1.0, delta=0.1, msg="Identical sentences should have BERTScore F1 near 1.0")

        candidate2 = "A completely unrelated statement."
        others2 = ["This is a test sentence."]
        # BERTScore F1 for very different sentences should be lower
        bert_f1_2 = get_bertscore(candidate2, others2)
        self.assertTrue(0.0 <= bert_f1_2 < 0.8, msg="Different sentences should have lower BERTScore F1") # BERTScore is generally higher than other metrics

        candidate3 = "This is a phrase for testing."
        others3 = ["This is a test sentence."]
        bert_f1_3 = get_bertscore(candidate3, others3)
        self.assertTrue(0.5 < bert_f1_3 <= 1.0, msg="Somewhat similar sentences should have intermediate BERTScore F1")

        # Test with empty others list
        candidate_empty_others = "Any sentence."
        others_empty = []
        bert_f1_empty = get_bertscore(candidate_empty_others, others_empty)
        self.assertEqual(bert_f1_empty, 0.0, msg="BERTScore with empty others list should be 0.0")

if __name__ == '__main__':
    unittest.main() 