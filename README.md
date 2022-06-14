## Project for ECE 285 course

In robotics, having a robot grasp an object in a cluttered environment is a challenging problem. With the advent of Convolutional Neural Networks (CNNs) has been proposed to tackle the problem of detecting the right grasp for an object. Several works also combine grasp detection and object detection to predict grasp candidates that are assigned to specific objects in the scene. In our project, we propose to use Detection Transformers(DETR) to perform grasp detection.

To train the network with Vanilla Loss:

python trainer_vanilla_loss.py

To train full DETR model with IoU Loss:

python trainer.py


Inference:

inference_Grasp_model.ipynb
