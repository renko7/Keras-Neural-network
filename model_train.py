from data_generation import *
from seq_model_generation import *

# will prepare the model before training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#trainings the model with 10 samples passed at ones 30 times
model.fit(x=scaled_train_samples, y=train_labels,validation_split=0.1, batch_size=10, epochs=30, verbose=2)
