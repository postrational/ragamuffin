import nltk
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity


class SemanticHighlighter:
    def __init__(self):
        # Load the embedding model
        model_name = "all-mpnet-base-v2"
        self.model = HuggingFaceEmbedding(model_name=model_name, embed_batch_size=32)
        # Download the NLTK tokenizer
        nltk.download("punkt_tab", quiet=True)

    def highlight_multiple(self, query: str, sources: list[str], max_length: int = 500) -> list[str]:
        """Highlight sentences in multiple source texts based on their similarity to the query.

        Args:
            query: The search query string.
            sources: List of source texts to highlight.
            max_length: Maximum length of the returned highlighted text for each source.

        Returns:
            List of HTML strings with highlighted sentences for each source.
        """
        # Split sentences from all sources and keep track of indices
        all_sentences = []
        source_sentence_indices = []
        start_idx = 0

        for source in sources:
            sentences = self._split_sentences(source)
            all_sentences.extend(sentences)
            end_idx = start_idx + len(sentences)
            source_sentence_indices.append((start_idx, end_idx))
            start_idx = end_idx

        if not all_sentences:
            return []

        # Encode query and all sentences together
        all_texts = [query, *all_sentences]
        embeddings = self._generate_text_embeddings(all_texts)
        query_embedding = embeddings[0]
        sentence_embeddings = embeddings[1:]
        similarities = cosine_similarity([query_embedding], sentence_embeddings).flatten()

        # Process each source separately
        results = []
        for start, end in source_sentence_indices:
            sentences_slice = all_sentences[start:end]
            similarities_slice = similarities[start:end]
            selected_indices = self._select_trimmed_sentences(sentences_slice, similarities_slice, max_length)
            result = self._apply_markup(sentences_slice, similarities_slice, selected_indices)
            results.append(result)

        return results

    def _generate_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the input texts."""
        return self.model.get_text_embedding_batch(texts)

    def _split_sentences(self, text: str) -> list[str]:
        """Tokenize the input text into sentences.

        Args:
            text: The text to split into sentences.

        Returns:
            A list of sentences.
        """
        return sent_tokenize(text)

    def _select_trimmed_sentences(self, sentences: list[str], similarities: np.ndarray, max_length: int) -> list[int]:
        """Select indices of sentences around the highest similarity sentence while respecting max_length.

        Args:
            sentences: List of sentences from the source text.
            similarities: Numpy array of similarity scores for each sentence.
            max_length: Maximum length of characters for the trimmed text.

        Returns:
            List of indices of selected sentences.
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

    def _apply_markup(self, sentences: list[str], similarities: np.ndarray, selected_indices: list[int]) -> str:
        """Apply HTML markup to the selected sentences.

        Args:
            sentences: List of all sentences.
            similarities: Numpy array of similarity scores for each sentence.
            selected_indices: List of indices of sentences to include.

        Returns:
            HTML string with marked-up sentences.
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
