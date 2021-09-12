# Learning-to-Use-ML
Here are all the programs that I have made in the pursuit of learning ML under the supervision of Invigilo AI.

**1. Headphone Detection Model**

This is the first headphone detection model, using which you can collect your own data, train the model, and run the program. It uses Mediapipe as a facial recognition algorithm to process the images, and uses Pytorch to run a simple CNN model. 

When you run the program, 4 options are presented:

1 and 2:
These take pictures of you from your webcam and store them in separate folders. You will have to alter the paths yourself.

3: 
This option processes the pictures from your folders and trains the model, saving the model in a separate folder.

4:
This option runs the latest model in your folder with models in it. If you wish to use a different model, you will have to change the path directly.

The program does not function with multiple people in frame, but it does not crash under those scenarios, or if no one is found in the frame.
