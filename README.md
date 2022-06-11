##This project for ECE 285 course

In robotics, having a robot grasp an object in a cluttered environment is a challenging problem. With the advent of Convolutional Neural Networks (CNNs) has been proposed to tackle the problem of detecting the right grasp for an object. Several works also combine grasp detection and object detection to predict grasp candidates that are assigned to specific objects in the scene. In our project, we propose to use Detection Transformers(DETR) to perform grasp detection.

To train vanilla network
python trainer_vanilla_loss.py

to train full detr model with iou loss
python trainer.py

infernce:
inference_Grasp_model.ipynb
