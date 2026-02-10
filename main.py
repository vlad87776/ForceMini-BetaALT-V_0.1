#!/usr/bin/env python3
"""
ForceMini 0.1βALT - Основной скрипт
"""

import argparse
import numpy as np
import os
import sys

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_engine import ForceMiniTokenizer, DataProcessor
from model_core import ForceMiniModel
from train_engine import ForceMiniTrainer

def prepare_sample_data():
    texts = [
        "Привет, как дела?",
        "Отличная погода сегодня",
        "Не очень хороший день",
        "Люблю программирование",
        "Ненавижу ошибки в коде",
        "Работа сделана хорошо",
        "Ужасный результат",
        "Превосходная работа",
        "Среднее качество",
        "Не могу дождаться выходных"
    ]
    
    labels = [1, 2, 0, 2, 0, 2, 0, 2, 1, 2]  # 0-негатив, 1-нейтрально, 2-позитив
    
    return texts, labels

def main():
    parser = argparse.ArgumentParser(description='ForceMini 0.1βALT - Тренировка модели')
    parser.add_argument('--mode', choices=['train', 'predict', 'test'], default='train')
    parser.add_argument('--model-path', type=str, default='models/forcemini_model.json')
    parser.add_argument('--vocab-path', type=str, default='vocab/forcemini_vocab.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=" * 60)
        print("ForceMini 0.1βALT - Тренировка модели")
        print("=" * 60)
        
        # Создаем папки если их нет
        os.makedirs('models', exist_ok=True)
        os.makedirs('vocab', exist_ok=True)
        
        # 1. Подготовка данных
        print("\n1. Подготовка данных...")
        texts, labels = prepare_sample_data()
        
        # 2. Токенизация
        print("2. Токенизация текстов...")
        tokenizer = ForceMiniTokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)
        tokenizer.save_vocab(args.vocab_path)
        
        # 3. Создание BOW векторов
        print("3. Создание признаков...")
        x_data = []
        for text in texts:
            seq = tokenizer.text_to_sequence(text)
            bow = DataProcessor.create_bow_vector(seq, len(tokenizer.word2idx))
            x_data.append(bow)
        
        x_data = np.array(x_data)
        
        # 4. One-hot кодирование меток
        y_data = DataProcessor.one_hot_encode(labels, num_classes=3)
        
        # 5. Разделение данных
        x_train, x_test, y_train, y_test = DataProcessor.train_test_split(x_data, y_data, test_size=0.3)
        
        # 6. Создание модели
        print("4. Создание модели...")
        vocab_size = len(tokenizer.word2idx)
        model = ForceMiniModel(
            input_size=vocab_size,
            hidden_size=64,
            output_size=3
        )
        
        # 7. Обучение
        print("5. Обучение модели...")
        trainer = ForceMiniTrainer(model, learning_rate=args.lr)
        
        history = trainer.train(
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=4,
            verbose=True
        )
        
        # 8. Сохранение модели
        print("6. Сохранение результатов...")
        trainer.save_model(args.model_path)
        
        print("\n✓ Обучение завершено!")
        print(f"Модель: {args.model_path}")
        print(f"Словарь: {args.vocab_path}")
    
    elif args.mode == 'test':
        print("=" * 60)
        print("Тестовый режим ForceMini")
        print("=" * 60)
        
        # Простой тест работы модели
        model = ForceMiniModel(input_size=10, hidden_size=5, output_size=3)
        
        # Тестовый forward pass
        test_input = np.random.randn(2, 10)
        output = model.forward(test_input)
        
        print(f"Входная форма: {test_input.shape}")
        print(f"Выходная форма: {output.shape}")
        print(f"Пример выхода (первые 2 строки):")
        print(output[:2])
        print(f"Сумма вероятностей в строках: {np.sum(output, axis=1)}")
        
        print("\n✓ Тест пройден успешно!")
    
    elif args.mode == 'predict':
        print("Режим предсказания будет реализован в следующей версии")
        print("Пока используйте режим 'train' для демонстрации")

if __name__ == "__main__":
    main()
