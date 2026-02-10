import numpy as np
import json

class ForceMiniTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.training_history = {'loss': [], 'accuracy': []}
    
    def train_step(self, x_batch, y_batch):
        predictions = self.model.forward(x_batch)
        loss = self.model.loss(predictions, y_batch)
        self.model.backward()
        self.model.update(self.learning_rate)
        return loss
    
    def compute_accuracy(self, predictions, labels):
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        return np.mean(pred_classes == true_classes)
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32, verbose=True):
        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            epoch_accuracies = []
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss = self.train_step(x_batch, y_batch)
                predictions = self.model.forward(x_batch)
                accuracy = self.compute_accuracy(predictions, y_batch)
                
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
            
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            
            if verbose:
                print(f"Эпоха {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
                
                if x_val is not None and y_val is not None:
                    val_predictions = self.model.forward(x_val)
                    val_loss = self.model.loss(val_predictions, y_val)
                    val_accuracy = self.compute_accuracy(val_predictions, y_val)
                    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                print("-" * 40)
        
        return self.training_history
    
    def save_model(self, filepath):
        model_data = {
            'weights': [],
            'biases': [],
            'architecture': self.model.get_config(),
            'training_history': self.training_history
        }
        
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                model_data['weights'].append(layer.weights.tolist())
                model_data['biases'].append(layer.bias.tolist())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Модель сохранена в {filepath}")
    
    @staticmethod
    def load_model(filepath, model_class):
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        model = model_class.from_config(model_data['architecture'])
        
        for i, (weights, bias) in enumerate(zip(model_data['weights'], model_data['biases'])):
            if i < len(model.layers):
                model.layers[i].weights = np.array(weights)
                model.layers[i].bias = np.array(bias)
        
        return model
