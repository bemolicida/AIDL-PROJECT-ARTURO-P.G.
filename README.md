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

#### TEST 1: LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|F|F|DECISION NET|77|76|T|54|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image6%20-%20test1%20loos.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image7%20-%20test1%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration presents overfit, we can see a separation between validation and training looses and a convexity at 30 epochs

#### TEST 2: LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITH DATA AUGMENTATION

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|F|T|DECISION NET|79|80|F|58|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image8%20-%20test2%20loos.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image9%20-%20test2%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration now doesn't present overfit, we have  a decent Validation accuracy and a nice test accuracy of 80%. As we will see in further combinations this one is the best candidate for a VGG with simple loss.


#### TEST 3: LR=5E-4, WD=0, N_EPOCHS=60 SIAMESE DECISION WITH DATA AUGMENTATION

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5E-4|0|T|16|T|F|T|DECISION NET|80|78|F|47|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image10%20-%20test3%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image11%20-%20test3%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration presents a nice validation accuracy and no overfit, by the other hand we can see that the accuracy is lower than previous test for testing sample, that lead us to preffer test2 for being more stable in that sense.


#### TEST 4: LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|T|T|DECISION NET|50|50|...|0|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image12%20-%20test4%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image13%20-%20test4%20accuracy.png?raw=true)]()

##### Conclussions:
In this execution we add freeze to the first 7 convolutional layers. The configuration presents strange values for losses and accuracies this is probably due to the fact that the learning rate is too big and the minimum loss is at a certain point that is inaccesible with an interval of 1e-3. We try in the next test to decrease this value to 5e-4 in order to verify our hipótesis. 

#### TEST 5: LR=5E-4, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5e-4|0|T|16|T|T|T|DECISION NET|81|79|T|56|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image14%20-%20test5%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image15%20-%20test5%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration with this lr solves our previous test problem and now we arribe to the minimum loss easely. The bad point is that we continue having overfit. 

#### TEST 6: LR=1E-5, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5e-4|0|T|16|T|T|T|DECISION NET|83|83|T|55|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image16%20-%20test6%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image17%20-%20test6%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration with the freeze and the learning rate of 1e-5 works great, we get an validation accuracy of 83 and a testing accuracy of 83 too. The problem that persists is the overfit that have been reflected in all tests with freeze that we have done.



