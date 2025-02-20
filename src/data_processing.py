from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import random

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    sorted_vocab: List[int] = sorted(word_counts,key=word_counts.get,reverse=True)
    
    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {}
    for i in range(len(sorted_vocab)):
        int_to_vocab[i] = sorted_vocab[i]

    vocab_to_int: Dict[str, int] = {}
    for i in range(len(sorted_vocab)):
        vocab_to_int[sorted_vocab[i]] = i



    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words.
    
    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

        
    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # TODO
    # Convert words to integers
    word_counts = Counter(words)
    total_words = len(words)
    word_freqs = {word: count / total_words for word, count in word_counts.items()}  # f(wi)

    # Step 2: Compute subsampling probabilities
    subsample_probs = {word: 1 - (threshold / freq) ** 0.5 for word, freq in word_freqs.items()}

    # Step 3: Subsample words based on probability
    int_words = [vocab_to_int[word] for word in words if word in vocab_to_int]  # Convert words to integers
    subsampled_words = [word_int for word_int in int_words if random.random() > subsample_probs.get(words[int_words.index(word_int)], 0)]

    return subsampled_words, word_freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    target_words: List[str] = []
    R = random.randint(1, window_size)

    # Define the start and end of the window, making sure we don't go out of bounds
    start_idx = max(idx - R, 0)
    end_idx = min(idx + R + 1, len(words))  # +1 because range() is exclusive

    # Get all words in the window except the target word itself
    target_words = [words[i] for i in range(start_idx, end_idx) if i != idx]

    return target_words


def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """

    # TODO
    pairs = []

    for idx, center_word in enumerate(words):
        R = random.randint(1, window_size)

        start_idx = max(idx - R, 0)
        end_idx = min(idx + R + 1, len(words))  

        # Extraer los contextos exxluyendo la palabra central
        for i in range(start_idx, end_idx):
            if i != idx: 
                pairs.append((center_word, words[i]))

    random.shuffle(pairs)

    # Generar batches
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        inputs, targets = zip(*batch) 
        yield list(inputs), list(targets)

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO
    valid_examples = torch.randint(0, valid_window, (valid_size // 2,), dtype=torch.long)
    valid_examples = torch.cat((valid_examples, torch.randint(len(embedding.weight) - valid_window, len(embedding.weight), (valid_size // 2,), dtype=torch.long)))
    valid_examples = valid_examples.to(device)

    valid_vectors = embedding(valid_examples)  

    embedding_weights = embedding.weight.data  
    embedding_weights = embedding_weights / embedding_weights.norm(dim=1, keepdim=True) 
    valid_vectors = valid_vectors / valid_vectors.norm(dim=1, keepdim=True)  

    similarities = torch.mm(valid_vectors, embedding_weights.T)  

    return valid_examples, similarities

