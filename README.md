# Estimation-of-Heart-rate-using-FastICA
Detection of heart rate through laptop webcam using Fast Independent Component Analysis (ICA). This project follows Thearn's (https://github.com/thearn/webcam-pulse-detector) work. Independent Component Analysis has been used to compute individual's heart beat through laptop camera. 

Photoplethysmography is the science of detecting vital signs through wavelength of light reflected from a person's face or skin. This project takes the visible light spectrum and computes hidden vital signs using Red, Green and Blue channels. Independent Component Analysis works on the principle of blind-source seperation. Using this we can seperate RGB channels into individual components, which later are used computed through fourier transform to estimate the person's heart-beat.

Flow Chart:

![flow_python](https://user-images.githubusercontent.com/39982386/47692115-b153d600-dbc2-11e8-9ecf-f380c70dfcb6.PNG)

Note: I'm using a basic laptop camera which clocks upto 29-30fps. Right now, the program approximates the hear-rate in 10 seconds. Increasing the fps would decrese the detection time. Remember, ICA stores data and performs the analysis. So, more data we can gather in little time the better the results. Have to test it using Xbox Kinect (good Depth-in Field and resolution). Testing in different light settings. Needs editing and proper identations.
