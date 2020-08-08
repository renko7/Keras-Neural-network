from data_generation import *
from seq_model_generation import *
from inference_data import *

# will prepare the model before training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#trainings the model with 10 samples passed at ones 30 times
#creates validation set from validation_split and runs after each epoch to check for overfitting
model.fit(x=scaled_train_samples, y=train_labels,validation_split=0.1, batch_size=10, epochs=30, verbose=2)


#checking the generalization of the model with new set of data "inference"
predictions = model.predict(x=scaled_test_samples,batch_size=10,verbose=0)


#returns which label with highest probability
rounded_predictions = np.argmax(predictions, axis=-1)

for i in rounded_predictions:
    print(i)

#creates a confusion matrix by passing the true labels and predicted labels. Creates plot labels for the labels
#will only work on jupiter notebook
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')