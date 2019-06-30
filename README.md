# AIDL-PROJECT-ARTURO-PALOMINO
ARTIFICIAL INTELLIGENCE AND DEEP LEARNING PROJECT - UPC 

Our first approach is to test different parameters of the optimizer, number of epochs, configuration of regularizations, and nets in order to arrive to the optimal combination that shows the best accuracy on validation and then with the choosen optimal combination make a test and check the test accuracy.

## Models


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



## Parameters


For the different tests we use the following parameters
![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image4%20-%20parameters.PNG?raw=true)



## Codes


We use two codes:

-One for the Siamesse Decision Network with single loss: "ARTUR PALOMINO CODE1.ipynb"

-One for the Siamesse Decision Network with two losses: "ARTUR PALOMINO CODE2.ipynb"


## Code  "ARTUR PALOMINO CODE1.ipynb"

We split the code in 7 sections:
MOUNTING DRIVE
↳ we mount google drive CFPW images that were previously uploaded 
IMPORTING PACKAGES
↳ we import different packages needed for the execution
CREATING DOWNLOADER CLASS
↳ The downloader class is used to feed the net with CFPW images
CREATING SIAMESE NETWORKS (VGG WITH DIFFERENT OPTIONS)
↳ We have a VGG decision network and VGG linear network (that last not used)
UTILITIES FOR METRICS OF OUR MODELS
↳ With this section we obtain the validation and training accuracy and losses
LOOP FOR TRAINING
↳ In this section we create the model, calculate the loss and backpropagate
LOOP FOR VALIDATION
↳ In this section the model is calculated with validation images

After this different tests are calculated with different parameter configurations


## Results for Code  "ARTUR PALOMINO CODE1.ipynb"

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image5%20-%20results%20code1.PNG?raw=true)

#### First Test

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|F|F|DECISION NET|77|76|T|54|
