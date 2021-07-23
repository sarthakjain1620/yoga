# yoga

I have created a model for yoga pose detection using opencv. I imported the pose estimation code for detecting key points in an image from github where it will generate a skeleton for a particular yoga pose.
<img width="960" alt="pose detected image with skeleton points" src="https://user-images.githubusercontent.com/83235872/126775888-90d42ecf-c891-473d-a9f1-804fc4a56635.png">

<img width="960" alt="pose detected image " src="https://user-images.githubusercontent.com/83235872/126775913-d6472ea6-6e96-4b89-8612-1084e48257a4.png">



Using it, I defined a function where it will go through the path in the train folder and it will go through each image and it will save each image with pose detected inside the sub folder. 
<img width="960" alt="cropped image folder with pose skeleton" src="https://user-images.githubusercontent.com/83235872/126775967-771ef980-c0bc-4143-a6ef-81b36a3f31ba.png">


After the train folder has the pose skeleton images for each pose, I labelled yoga pose name into numbers.for eg: downdog:0,goddess:1.

I created X and y for the model and then split it using train test split in X_train,y_train,X_test,y_test.

I used svm pipeline and supplied X_train and y_train for training the model.

Using X_test and y_test, I predicted the yoga poses. Here, I achieved 75% accuracy on the train folder.
<img width="960" alt="accuracy on train folder" src="https://user-images.githubusercontent.com/83235872/126775982-62f90598-60f2-4746-9dd1-08b381a51e76.png">


After training, I saved the model using a pickle file.

Then, I load the model again for the test folder.

Predicting on the test folder, I achieved 70% accuracy.
<img width="960" alt="accuracy on test folder" src="https://user-images.githubusercontent.com/83235872/126776006-bdbca9c8-d04c-4c7b-8c41-206fc47d9720.png">

Drive Link-https://drive.google.com/drive/folders/1FdU7r81t_g0PpLEiJuSppxPGhftXvABu?usp=sharing
The above link contains images of train folder and test folder with each image containing the points detected by the function.

