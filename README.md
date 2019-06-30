# AIDL-PROJECT-ARTURO-PALOMINO
ARTIFICIAL INTELLIGENCE AND DEEP LEARNING PROJECT - UPC 

Our first approach is to test different parameters of the optimizer, number of epochs, configuration of regularizations, and nets in order to arrive to the optimal combination that shows the best accuracy on validation and then with the choosen optimal combination make a test and check the test accuracy.

### Models


For our exercise we have tested two net configurations:
- Siamesse Decision Network with simple loss. traditional siamese net where two nets are calculated in paralel, the results are concatenated and the resulting loss is used to backpropagate
- Siamesse Decision Network with the average of two losses. A different approach in which first we feed the siamese net with the positive (true same person pairs) cases and we obtain its loss, then we feed again the siamese net with negative cases (non same person pairs). Then the average of both losses is calculated and used to backpropagate.
![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/image1%20-%20siamese.png?raw=true)

*Image  from Amazon [Amazon](https://aws.amazon.com/es/blogs/machine-learning/combining-deep-learning-networks-gan-and-siamese-to-generate-high-quality-life-like-images/)

For Each branch we choose two Pretrained Nets:
- VGG
- Alexnet

Detail of the net structure
- VGG:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image2%20-%20vgg16.png?raw=true)

- ALEXNET

*Image  from Neurohive [Neurohive](https://neurohive.io/en/popular-networks/vgg16/)
![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image3%20-%20AlexNet-CNN-architecture-layers.png?raw=true)

*Image  from Researchgate [Researchgate](https://www.researchgate.net/figure/AlexNet-CNN-architecture-layers_fig1_318168077)



### Parameters


For the different tests we use the following parameters
![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image4%20-%20parameters.PNG?raw=true)

