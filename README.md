# yoga

I have created a model for yoga pose detection using opencv. I imported the pose estimation code from github where it will generate a skeleton for a particular yoga pose.

Using it, I defined a function where it will go through the path in the train folder and it will go through each image and it will save each image with pose detected inside the sub folder. 

After the train folder has the pose skeleton images for each pose, I labelled yoga pose name into numbers.for eg: downdog:0,goddess:1.

I created X and y for the model and then split it using train test split in X_train,y_train,X_test,y_test.

I used svm pipeline and supplied X_train and y_train for training the model.

Using X_test and y_test, I predicted the yoga poses. Here, I achieved 75% accuracy on the train folder.

After training, I saved the model using a pickle file.

Then, I load the model again for the test folder.

Predicting on the test folder, I achieved 70% accuracy.
