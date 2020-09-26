# Sonar to image by cGAN
An implementation* of sonar to image in Keras.

*K. Terayama, K. Shin, K. Mizuno, K. Tsuda, "Integration of sonar and optical camera images using deep neural network for fish monitoring," Aquacultural Engineering, Vol. 86, 102000, 2019. [DOI: 10.1016/j.aquaeng.2019.102000]

## Requirements
- python 3.6.1  
- NumPy 1.13.3  
- tensorflow 1.2.1  
- Keras 2.0.6  
- Pillow 4.2.1  

## USAGE
- train without sonar images
`python train.py -g [gpu ID] -b [batch size] -e [number of epochs] -o [output directory] -l [lambda] -d [darkness]`
- train with sonar images
`python train_sonar.py -g [gpu ID] -b [batch size] -e [number of epochs] -o [output directory] -l [lambda] -d [darkness]`


## METHOD
![network](https://user-images.githubusercontent.com/17425130/35025664-155e67ec-fb8a-11e7-9e98-697c07d5b163.png)
