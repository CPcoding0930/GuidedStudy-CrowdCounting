# GuidedStudy-CrowdCounting

## Citation
#### CSRNet Model (Pytorch)
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1091--1100},
  year={2018}
}

#### Dataset (CityStreet)
@inproceedings{zhang2019wide,
title={Wide-Area Crowd Counting via Ground-Plane Density Maps and Multi-View Fusion CNNs},
author={Zhang, Qi and Chan, Antoni B},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
pages={8297–8306},
year={2019}
}
@inproceedings{zhang2020wide,
title={Wide-Area Crowd Counting: Multi-View Fusion Networks for Counting in Large Scenes},
author={Zhang, Qi and Chan, Antoni B},
booktitle={https://arxiv.org/abs/2012.00946},
year={2020}
}

## Project Description
Crowd Counting is a common and practical measurement nowadays. The crowd counting
is defined as the technique to estimate the number of detected people. As humans cannot
accurately measure the number of people in a detected scene quickly, especially when a
tremendous amount of people is crammed inside a screen, the use of deep learning and
computer vision can greatly increase the efficiency and reduce the error rate. Under the
development of computer vision related technology, the use of convolutional neural networks
(CNN) would be useful to train the model to learn the characteristics of images in order to
generate a density map, which is useful in predicting and estimating the crowd number. This
project adapts this state-of-art network method - Dilated Convolutional Neural Networks -
with three steps: generate ground truth density map using the given dataset; train the model
with data; make the prediction and test the results. To go through the entire application, this
paper depicts from the background of the project, including the comparison of possible
approaches, to the design and implementation, with the help of the dataset CityStreet
provided by the City University of Hong Kong.

This also serves as my first computer vision related project. 

## Project Implementation
Adoption on CRSNet on CiyuStreet. 
1. Run the ipynb to train the image
2. Change the corresponding json folder
3. Run the train.py with the json files
4. After training model, test with the val.ipynb to get the MSE and predicted counts.

Results as:
![image](https://user-images.githubusercontent.com/59085118/115141712-679c2600-a070-11eb-8379-d67b46577574.png)

## Project Improvement
This project is set to be the pre-version of the social distancing violation project, including the use of density map and camera project. 
Due to the time limitation, it has not be fully implement, yet the direction includes but not limits to:
1. Use the camera distance to calculate the size of the image patch in real meters, which is the transformation between images to reality;
2. Estimate the people count from the density map in the given image patch;
3. Calcualte the people density in count per meter square from the people count and image patch size; 
4. Threshold the people density accordingly to get the social distance violation map. 

## Project Structure
![image](https://user-images.githubusercontent.com/59085118/115141952-b0081380-a071-11eb-9c82-ee4f20fd0552.png)
