import re
import numpy as np
import json

class ForceMiniTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.frequencies = {}
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def build_vocab(self, texts):
        self.frequencies = {}
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                self.frequencies[token] = self.frequencies.get(token, 0) + 1
        
        sorted_words = sorted(self.frequencies.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        
        words = [word for word, _ in sorted_words[:self.vocab_size - len(self.special_tokens)]]
        all_words = self.special_tokens + words
        
        self.word2idx = {word: idx for idx, word in enumerate(all_words)}
        self.idx2word = {idx: word for idx, word in enumerate(all_words)}
    
    def text_to_sequence(self, text, max_length=None):
        tokens = self.preprocess_text(text)
        sequence = [self.word2idx.get(self.SOS_TOKEN, 0)]
        
        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx.get(self.UNK_TOKEN, 1))
            sequence.append(idx)
        
        sequence.append(self.word2idx.get(self.EOS_TOKEN, 3))
        
        if max_length:
            if len(sequence) < max_length:
                sequence += [self.word2idx.get(self.PAD_TOKEN, 0)] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        
        return sequence
    
    def save_vocab(self, filepath):
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = {k: int(v) for k, v in vocab_data['word2idx'].items()}
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.vocab_size = vocab_data['vocab_size']

class DataProcessor:
    @staticmethod
    def create_bow_vector(sequence, vocab_size):
        vector = np.zeros(vocab_size)
        for idx in sequence:
            if idx < vocab_size:
                vector[idx] += 1
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        return vector
    
    @staticmethod
    def one_hot_encode(labels, num_classes):
        encoded = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            encoded[i, label] = 1
        return encoded
    
    @staticmethod
    def train_test_split(x_data, y_data, test_size=0.2, random_seed=42):
        np.random.seed(random_seed)
        n_samples = x_data.shape[0]
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        return x_data[train_idx], x_data[test_idx], y_data[train_idx], y_data[test_idx]
