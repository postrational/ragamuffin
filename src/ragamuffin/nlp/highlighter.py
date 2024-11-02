import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticHighlighter:
    def __init__(self) -> None:
        # Load the embedding model
        model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
        # Download the NLTK tokenizer
        nltk.download("punkt", quiet=True)

    def highlight_text(self, query: str, source: str, max_length: int = 500) -> str:
        """Main method to highlight sentences in the source based on similarity to the query.

        Args:
        - query: The search query string.
        - source: The source text to highlight.
        - max_length: Maximum length of the returned highlighted text.

        Returns:
        - HTML string with highlighted sentences based on similarity.

        """
        sentences = self._split_sentences(source)
        similarities = self._compute_similarities(query, sentences)
        selected_indices = self._select_trimmed_sentences(sentences, similarities, max_length)
        return self._apply_markup(sentences, similarities, selected_indices)

    def _split_sentences(self, text: str) -> list[str]:
        """Tokenize the input text into sentences."""
        return sent_tokenize(text)

    def _compute_similarities(self, query: str, sentences: list[str]) -> np.ndarray:
        """Compute cosine similarity between query and each sentence in the list.

        Args:
        - query: Query string to compare against each sentence.
        - sentences: List of sentences to compute similarity with.

        Returns:
        - Numpy array of similarity scores.

        """
        query_embedding = self.model.encode([query])
        sentence_embeddings = self.model.encode(sentences)
        return cosine_similarity(query_embedding, sentence_embeddings)[0]

    def _select_trimmed_sentences(self, sentences: list[str], similarities: np.ndarray, max_length: int) -> list[int]:
        """Select indices of sentences around the highest similarity sentence while respecting max_length.

        Args:
        - sentences: List of sentences from the source text.
        - similarities: Numpy array of similarity scores for each sentence.
        - max_length: Maximum length of characters for the trimmed text.

        Returns:
        - List of indices of selected sentences.

        """
        highest_sim_index = int(np.argmax(similarities))
        selected_indices = [highest_sim_index]
        char_count = len(sentences[highest_sim_index])

        left = highest_sim_index - 1
        right = highest_sim_index + 1

        while (left >= 0 or right < len(sentences)) and char_count < max_length:
            next_index = None

            # Decide whether to pick left or right sentence
            if left >= 0 and right < len(sentences):
                # Choose the side with the higher similarity
                if similarities[left] >= similarities[right]:
                    next_index = left
                    left -= 1
                else:
                    next_index = right
                    right += 1
            elif left >= 0:
                next_index = left
                left -= 1
            elif right < len(sentences):
                next_index = right
                right += 1

            if next_index is not None:
                sentence_length = len(sentences[next_index])
                if char_count + sentence_length > max_length:
                    break
                selected_indices.append(next_index)
                char_count += sentence_length
            else:
                break

        # Sort the indices to maintain the original order
        selected_indices.sort()
        return selected_indices

    def _apply_markup(
        self,
        sentences: list[str],
        similarities: np.ndarray,
        selected_indices: list[int],
    ) -> str:
        """Apply HTML markup to the selected sentences.

        Args:
        - sentences: List of all sentences.
        - similarities: Numpy array of similarity scores for each sentence.
        - selected_indices: List of indices of sentences to include.

        Returns:
        - HTML string with marked-up sentences.
        """
        marked_up_sentences = []
        for idx in selected_indices:
            sentence = sentences[idx]
            similarity = similarities[idx]
            similarity_class = int(min(similarity * 10, 9))
            marked_up_sentences.append(f'<span class="similarity-{similarity_class}">{sentence}</span>')

        marked_text = " ".join(marked_up_sentences)

        # Add ellipses if trimmed text excludes start or end of the source
        if selected_indices[0] > 0:
            marked_text = "..." + marked_text
        if selected_indices[-1] < len(sentences) - 1:
            marked_text += "..."
        return marked_text
