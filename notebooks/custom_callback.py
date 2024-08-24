from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np

class F1Checkpoint(Callback):
    def __init__(self, filepath, validation_data, monitor='val_f1_score', mode='max', verbose=1, save_best_only=True, save_weights_only=False):
        super(F1Checkpoint, self).__init__()
        self.filepath = filepath
        self.validation_data = validation_data
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_f1 = -np.inf if mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        # Get validation data
        if isinstance(self.validation_data, tuple):
            val_data, val_labels = self.validation_data
        else:
            # Get data from all batches and concatenate if it is ImageDataGenerator
            val_data, val_labels = [], []
            for batch in self.validation_data:
                val_data.append(batch[0])
                val_labels.append(batch[1])
            val_data = np.concatenate(val_data, axis=0)
            val_labels = np.concatenate(val_labels, axis=0)

        val_pred = self.model.predict(val_data)
        val_pred = np.argmax(val_pred, axis=1)
        val_labels = np.argmax(val_labels, axis=1)

        # Calculate F1-score
        current_f1 = f1_score(val_labels, val_pred, average='macro')

        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: Validation F1-score: {current_f1}')

        # Check if we should save the model
        if self.save_best_only:
            if (self.mode == 'max' and current_f1 > self.best_f1) or (self.mode == 'min' and current_f1 < self.best_f1):
                self.best_f1 = current_f1
                if self.save_weights_only:
                    self.model.save_weights(self.filepath)
                else:
                    self.model.save(self.filepath)
                if self.verbose > 0:
                    print(f'F1-score improved. Saving model to {self.filepath}')
        else:
            if self.save_weights_only:
                self.model.save_weights(self.filepath)
            else:
                self.model.save(self.filepath)
            if self.verbose > 0:
                print(f'Saving model to {self.filepath}')
