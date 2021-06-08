# Classifying RGB images
We are going to train and optimize an artificial neural network on images from the CIFAR database. In that case, the images are classified according to 10 different classes : airplane, car, truck, bird, cat, dog, deer, horse, frog and ship. <br>

There are several difficulties : <br>

- Several classes are quite close (such as car and truck) and will require a thourough training of the network to efficiently differentiate the images. <br>
- The images are RGB (and not grayscale as in the dataset MNIST) and of very low quality.<br>

From the root of the project (Classifying-RGB-images/): <br>
``` $ pip3 install -r requirements.txt ``` <br>
``` $ python3 main.py ```<br>

