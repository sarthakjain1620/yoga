#!/usr/bin/env python
# coding: utf-8

# # Yoga Pose Detection Model

# In[1]:


import cv2
import time


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# # imported weights

# In[3]:


net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")#weights


# In[4]:


inWidth=368
inHeight=368
thr=0.2


# In[5]:


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


# In[6]:


imgg=cv2.imread("./Train/downdog/00000133.jpg")
imgg.shape


# In[7]:


plt.imshow(imgg)


# In[8]:


plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))


# # function pose estimation to detect key points in an image(imported the code from github)

# In[9]:


def pose_estimation(imgs):
    imgsWidth=imgs.shape[1]
    imgsHeight=imgs.shape[0]
    net.setInput(cv2.dnn.blobFromImage(imgs,1.0,(inWidth,inHeight),(127.5,127.5,127.5),swapRB=True,crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (imgsWidth * point[0]) / out.shape[3]
        y = (imgsHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(imgs, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(imgs, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(imgs, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(imgs, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return imgs


# # function to read all images from the image path in Train folder and detect poses for the same

# In[10]:


def estimated_image(image_path):
    img = cv2.imread(image_path)
    img_s=img
    if img is not None:
        img_s=pose_estimation(img)
    return img_s


# In[11]:


i=estimated_image("./Train/goddess/00000101.jpg")


# # an example of a pose detection

# In[12]:


plt.imshow(i)


# In[13]:


estimat_image=pose_estimation(imgg)


# In[14]:


plt.imshow(cv2.cvtColor(estimat_image, cv2.COLOR_BGR2RGB))


# In[71]:


path_to_data = "./Train/"
path_to_cr_data = "./Train/cropped/"


# In[72]:


import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)


# # here I created sub directories within the train folder of the points detected on each image

# In[73]:


img_dirs


# In[74]:


import shutil
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)


# # It ran through all images in the train folder and saved the pose detected image in the sub cropped folder

# In[75]:


cropped_image_dirs = []
yoga_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    yoga_name = img_dir.split('/')[-1]
    yoga_file_names_dict[yoga_name] = []
    for entry in os.scandir(img_dir):
        img_s = estimated_image(entry.path)
        if img_s is not None:
            cropped_folder = path_to_cr_data + yoga_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
            cropped_file_name = yoga_name + str(count) + ".jpg"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, img_s)
            yoga_file_names_dict[yoga_name].append(cropped_file_path)
            count += 1


# # Manually examine cropped folder and delete any unwanted images 

# In[76]:


yoga_file_names_dict = {}
for img_dir in cropped_image_dirs:
    yoga_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    yoga_file_names_dict[yoga_name] = file_list
yoga_file_names_dict


# # I labelled in numbers for each yoga pose name

# In[77]:


class_dict = {}
count = 0
for yoga_name in yoga_file_names_dict.keys():
    class_dict[yoga_name] = count
    count = count + 1
class_dict


# # created x and y to be supplied to the model

# In[78]:


X, y = [], []
for yoga_name, training_files in yoga_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1)))
        X.append(combined_img)
        y.append(class_dict[yoga_name])     


# In[79]:


len(X[0])


# In[80]:


X[0]


# In[81]:


y[0]


# In[83]:


X = np.array(X).reshape(len(X),3072).astype(float)
X.shape


# In[84]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# # created training and testing sample within train folder and supply it to the svm model.

# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[86]:


print(classification_report(y_test, pipe.predict(X_test)))


# In[87]:


y_predicted=pipe.predict(X_test)


# # Here I achieved 75% accuracy with the training sample using svm.

# In[88]:


print(classification_report(y_test,y_predicted))


# # save the model using joblib and pickle

# In[90]:


get_ipython().system('pip install joblib')
import joblib 
# Save the model as a pickle in a file 
joblib.dump(pipe, 'saved_model3.pkl')


# In[91]:


Pkl_Filename = "Pickle_pipe_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(pipe, file)


# # load the model again for testing in test folder

# In[92]:


joblib_pipe_model = joblib.load("saved_model3.pkl")


joblib_pipe_model


# In[93]:


with open(Pkl_Filename, 'rb') as file:  
    Pickled_pipe_Model = pickle.load(file)

Pickled_pipe_Model


# In[33]:


y_predicted[:5]


# In[34]:


y_test[:5]


# In[35]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# In[36]:


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}


# # Also before testing it into test folder I check whether any other model will perform better than svm. But I got the highest accuracy with the svm model and so I proceed it with it.

# In[37]:


scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[38]:


best_estimators['svm'].score(X_test,y_test)


# In[39]:


best_estimators['random_forest'].score(X_test,y_test)


# In[40]:


best_estimators['logistic_regression'].score(X_test,y_test)


# In[41]:


best_clf = best_estimators['svm']


# In[42]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
cm


# In[43]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[44]:


class_dict


# # I repeat the steps for the test folder to create x and y samples to supply it into the loaded model

# In[46]:


path_to_data = "./Test/"
path_to_cr_data = "./Test/cropped/"


# In[48]:


import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)


# In[49]:


img_dirs


# In[50]:


import shutil
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)


# # pose points detected in test folder

# In[51]:


cropped_image_dirs = []
yoga_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    yoga_name = img_dir.split('/')[-1]
    yoga_file_names_dict[yoga_name] = []
    for entry in os.scandir(img_dir):
        img_s = estimated_image(entry.path)
        if img_s is not None:
            cropped_folder = path_to_cr_data + yoga_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
            cropped_file_name = yoga_name + str(count) + ".jpg"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, img_s)
            yoga_file_names_dict[yoga_name].append(cropped_file_path)
            count += 1


# In[52]:


yoga_file_names_dict = {}
for img_dir in cropped_image_dirs:
    yoga_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    yoga_file_names_dict[yoga_name] = file_list
yoga_file_names_dict


# In[53]:


class_dict = {}
count = 0
for yoga_name in yoga_file_names_dict.keys():
    class_dict[yoga_name] = count
    count = count + 1
class_dict


# # Test samples created for the test folder as X1 and y1

# In[57]:


X1, y1 = [], []
for yoga_name, training_files in yoga_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1)))
        X1.append(combined_img)
        y1.append(class_dict[yoga_name])


# In[58]:


len(X1[0])


# In[59]:


X1 = np.array(X1).reshape(len(X1),3072).astype(float)
X1.shape


# # Supply X1 and y1 to the loaded model using both pickle and joblib

# In[94]:


score = Pickled_pipe_Model.score(X1, y1)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_pipe_Model.predict(X1)  

Ypredict


# In[95]:


score = joblib_pipe_model.score(X1, y1)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Y_predict = joblib_pipe_model.predict(X1)  

Y_predict


# # I check for some samples where I found it is showing error for some values while working fine for most cases

# In[100]:


Ypredict[10:20]


# In[101]:


y1[10:20]


# # I achieved 70% accuracy with the test folder and works fine for the majority of the test cases

# In[102]:


print(classification_report(y1,Ypredict))

