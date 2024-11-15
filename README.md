+ Use python 3.7 to build.
+ Load necessary libraries mentioned in requirements.txt.
+ Some requirements may not be mentioned, follow the interpret instruction to install them.
+ When you are all set, run the train_mask_detector.py to train the model.
+ When model is available, run face_mask_detection.py to get in to the GUI.


**** 
## Example GUI
![img.png](GUI.png)

## Evaluation Metrics
### LR=0.0001 EPOCHS=20 DROPOUT=0.5
![img.png](Metrics1.png)
### LR=0.0005 EPOCHS=20 DROPOUT=0.6
![img.png](Metrics2.png)
### LR=0.0001 EPOCHS=30 DROPOUT=0.5
![img.png](Metrics3.png)

## Loss&Accuracy Plot
### LR=0.0001 EPOCHS=20 DROPOUT=0.5
![Loss_Accuracy_LR_0.0001_EPOCHS_20.png](Loss_Accuracy_LR_0.0001_EPOCHS_20.png)
### LR=0.0005 EPOCHS=20 DROPOUT=0.6
![Loss_Accuracy_LR_0.0005_EPOCHS_20.png](Loss_Accuracy_LR_0.0005_EPOCHS_20.png)
### LR=0.0001 EPOCHS=30 DROPOUT=0.5
![Loss_Accuracy_LR_0.0001_EPOCHS_30.png](Loss_Accuracy_LR_0.0001_EPOCHS_30.png)