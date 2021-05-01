# trAIner24
 
## 1. Business Question
This project aims to detect the users’ motion when doing exercises, help them count or time the exercise, and tell them whether their exercise forms are correct or not. 

## 2. Model Architectures
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00006.jpg" width = "500">

## 3. OpenPose
OpenPose is the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints on single images. It is widely used for pose estimation, and also action recognition, which is what trAIner24 is going to do, recognizing correct and incorrect forms of workout.\
<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image25.png" width = "500">

### Two-branch Multi-stage CNN
<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image27.png" width = "500">
For the operation of OpenPose, an RGB image is firstly fed as input into a “two-branch multi-stage” CNN. Two branch means that the CNN produces two different outputs, and multi-stage simply means that the network is stacked one on the other at every stage.

### Confidence Maps
<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image28.png" width = "500">
The top branch, predicts the confidence maps of different body parts location such as the knees, elbows and others. 

### Affinity Fields
<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image29.png" width = "500">
The bottom branch predicts the affinity fields, which represent a degree of association between different body parts. \
\
As for multi-stage, the predictions in the previous stage, along with the original image features, are concatenated in each subsequent stage to produce more refined predictions.
\
Finally, the confidence maps and affinity fields are being processed to output the 2D key points for all people in the image.

## 4. Action Recognition Models
We trained our own action recognition models, one for squat and one for plank. For the data, we tried our best and took several videos for squat and plank, and then changed them into images as data input. There are in total 2659 images for squat. And for plank, there are 5611 images.\
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00011.jpg" width = "500">

For the action recognition process, we first input the frame (i.e. the images mentioned before), and then get the key points by OpenPose. After that, we preprocess the key points, through normalizing them with the person’s height, and obtaining velocity of action by comparing coordinates between frames. We obtain the time-serial features using the speed, and normalized positions from multiple adjacent images. Next, we used PCA to reduce the feature dimension from 314 to 50. Then, we classify correct and incorrect actions by a DNN model with 3 layers of 20, 30 and 50 neurons on each layer. Finally, we output the label and skeleton to the output frame, save the frames to videos, and show them on the result page. \
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00012.jpg" width = "500">

### Action Recognition Models Accuracy
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00013.jpg" width = "500">

### Action Recognition Demo: Squat
The following are the action recognition demonstrations for squat. \

<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image37.gif" width = "500">\
The first one shows a correct squat, where the person starts with the feet wider than the hip-width and focus on turning the knees out when going down. \

<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image38.gif" width = "500">\
The second one shows an incorrect squat, where the knees are turning inward. 

### Action Recognitio Demo: Plank
<img src="https://github.com/eddylamhw/trAIner24/blob/main/ppt/media/image39.gif" width = "500">
This is the demonstration for plank, it considers the form as incorrect when the hip is too low or too high.

## 5. Counting Recognition Model
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00017.jpg" width = "500">
The optical flow algorithm is first used to preprocess the incoming images, because it can compare two adjacent images, and recognizes upward motion with purple colour, downward with green colour, and static with no colour.\
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00018.jpg" width = "500">
After adopting the optical flow algorithm, the model can easily classify upward, downward, and static with an accuracy of 100%.

## 6. Deployment
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00020.jpg" width= "500">
For deployment, we used Flask to deploy the python app to HTML and used CSS to design the webpage. The users will see three webpages, the index page where the users can select whether he wants to do squat or plank. Once the user made the decision, he will see the squat or plank page. After that, he will either submit the number of squats or the duration of plank, then the webcam will start recording the video and the the workout detection will begin after 3 seconds. After finishing the workout, the user can then click the show result button and go to the result page, where he can see the result of his workout, and the videos for him to review why he had performed the exercise correctly or incorrectly. 

## 7. Demo
### Squat
Please download the "SquatDemo.mov" file to view the demo, as the video size is too large to be displayed in GitHub.

### Plank
Please download the "PlankDemo.mov" file to view the demo, as the video size is too large to be displayed in GitHub.

## 8. Challenges and Further Improvement
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00025.jpg" width= "500">
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00026.jpg" width= "500">

# Reference
<img src="https://github.com/eddylamhw/trAIner24/blob/main/images(ppt)/capstone_project_00027.jpg" width= "500">
The codes for OpenPose and action recognition models are largely compiled by felixchenfy, who originally used them to classify 9 actions, including jump, kick, punch, run, sit, squat, stand, walk, and wave. We mainly used his codes for classifying correct and incorrect actions, and then deployed the program as a Flask app.

