# AIDL-PROJECT-ARTURO-PALOMINO
ARTIFICIAL INTELLIGENCE AND DEEP LEARNING PROJECT - UPC 


## Introduction
Student: Arturo Palomino
Team: 4
Results of the final project of the Deep Learning and Artificial Intelligence post degree of the UPC talent-school.

For this project different architectures have been tested in order to train a model able to recognize if two faces correspond to the same person. For this purpose we use Siamese networks that returns a classification output with a binary answer, in our case the answer is a vector of two elements. If the first element of the vector is higher than the second element then we can say that the person is the same, in other cases the person is not the same.

Our first approach is to test different parameters of the optimizer, number of epochs, configuration of regularizations, and nets in order to arrive to the optimal combination that shows the best accuracy on validation and then with the chosen optimal combination make a test and check the test accuracy.


## Models


For our exercise we have tested two net configurations:
- Siamese Decision Network with simple loss. traditional Siamese net where two nets are calculated in parallel, the results are concatenated, and the resulting loss is used to backpropagate
- Siamese Decision Network with the average of two losses. A different approach in which first we feed the Siamese net with the positive (true same person pairs) cases and we obtain its loss, then we feed again the Siamese net with negative cases (non same person pairs). Then the average of both losses is calculated and used to backpropagate. In this way we ensure that exactly the 50% of the cases are of one case and 50% of the opposite, this mechanism helps the net to train better in the opinion of the author of this project.


![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/image1%20-%20siamese.png?raw=true)

*Image  from Amazon [Amazon](https://aws.amazon.com/es/blogs/machine-learning/combining-deep-learning-networks-gan-and-siamese-to-generate-high-quality-life-like-images/)

For Each branch we choose two Pretrained Nets:
- VGG
- Alexnet

Detail of the net structures:

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


We use only two colab codes:

-One for the Siamesse Decision Network with single loss: "ARTUR PALOMINO CODE1.ipynb"

-One for the Siamesse Decision Network with two losses: "ARTUR PALOMINO CODE2.ipynb"


## Code  "ARTUR PALOMINO CODE1.ipynb" Structure

This code is for testing different configurations for a VGG with single loss.

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





## Code  "ARTUR PALOMINO CODE2.ipynb" Structure

This code is for testing different configurations for a VGG and an Alexnet with two losses.

We split the code in the following sections:

MOUNTING DRIVE

↳ we mount google drive CFPW images that were previously uploaded 

IMPORTING PACKAGES

↳ we import different packages needed for the execution

CREATING DOWNLOADER CLASS

↳ The downloader class is used to feed the net with CFPW images

CREATING MODELS

↳ We have a VGG decision network and Alexnet with two losses.

TRAIN

↳ training function

TEST

↳ test function

MAIN

↳ main function

DOWNLOAD FILES (JUST IN CASE)

↳ download files from original websit in case needed

After this different tests are calculated with different parameter configurations


## Results for Code  "ARTUR PALOMINO CODE2.ipynb"

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image18%20-%20results%20code2.PNG?raw=true)

#### TEST A: ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|F|DECISION NET 2 LOSS|81|80|T|15|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image19%20-%20testA%20loss.png?raw=true)]()

##### Conclussions:
The configuration is not optimal as it shows overfit. The accuracy is nice and we can see the the optimal accuracy is achieved fast, in less than 10 epochs. Another interesting thing is that this net trains 60 epochs in les than 60 minutes while VGG lasts 4 times more.


#### TEST B: ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|T|DECISION NET 2 LOSS|84|82|F|59|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image20%20-%20testA%20loss.png?raw=true)]()

##### Conclussions:
The configuration doesn't show overfit moreover we arrive to an outstanding validation accuracy of 84 and a test accuracy of 82, this is by far the best combination until that moment. This is in fact one of the best options for Alexnet in this document.


#### TEST C: VGG, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|F|F|F|DECISION NET 2 LOSS|80|78|F|31|

##### Loss image & Accuracy image:
[![N|Solid](https://raw.githubusercontent.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/master/images/Image21%20-%20testC%20loss.png?raw=true)]()

##### Conclussions:
The configuration using a VGG shows worst results than the Alexnet in the same conditions, without data augmentation, without pretraining and without freeze, by the other hand the model has overfit.

#### TEST D: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|F|F|DECISION NET 2 LOSS|84|83|T|5|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image22%20-%20testD%20loss.png?raw=true)]()

##### Conclussions:
The configuration using a VGG and pretrain we obtain a nice validation accuracy of 84 and a test accuracy of 83, better than the last model except for the fact that this one has overfit.


#### TEST E: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|F|T|DECISION NET 2 LOSS|86|84|T|15|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image23%20-%20testE%20loss.png?raw=true)]()

##### Conclussions:
The configuration shows the best validation accuracy of the whole set of combinations, an 86%. Still have some overfit but validation curve increases gradually. It is probably the best option for the VGG with two losses.


#### TEST F: VGG, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

(No image)

##### Conclussions:
In order to test the learning rate 1e-3 we run again our last model, the results are the worst until the moment, the problem is that the accuracy never improves from 50%, the reason is that it's not possible for the model to arrive to the minimum loss beacuse the interval at every step is too big and the minimum is allways in the middle of the last two steps.




#### TEST G: VGG, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5E-4|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image25%20-%20testG%20loss.png?raw=true)]

##### Conclussions:
In order to test again the model, the learning rate is fixed at 5e-4. We find exactly the same scenario as in testG.



#### TEST H: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|T|T|DECISION NET 2 LOSS|84|84|T|16|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image26%20-%20testH%20loss.png?raw=true)]

##### Conclussions:
In this combination we include the option freeze, freezing 7 convolutions of the pretrained layers on the VGG. The test accuracy shows the best results of the table, the problem again is the overfit


#### TEST I: ALEXNET, LR 5E-4 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|T|DECISION NET 2 LOSS|83|81|F|239|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image27%20-%20testI%20loss.png?raw=true)]

##### Conclussions:
In this combination we run more epochs for the alexnet of the Test B. As we can see, the combination still don't have overfit, is quite robust. The only problem is that we don't improve the accuracy level of Test B.




#### TEST J: ALEXNET, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|5E-4|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image28%20-%20testJ%20loss.png?raw=true)]

##### Conclussions:
In this combination similarly to the case of test F and G we don't have convergency to the minimum of the loss




#### TEST K: ALEXNET, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-3|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

# AIDL-PROJECT-ARTURO-PALOMINO
ARTIFICIAL INTELLIGENCE AND DEEP LEARNING PROJECT - UPC 


## Introduction
Student: Arturo Palomino
Team: 4
Results of the final project of the Deep Learning and Artificial Intelligence post degree of the UPC talent-school.

For this project different architectures have been tested in order to train a model able to recognize if two faces correspond to the same person. For this purpose we use Siamese networks that returns a classification output with a binary answer, in our case the answer is a vector of two elements. If the first element of the vector is higher than the second element then we can say that the person is the same, in other cases the person is not the same.

Our first approach is to test different parameters of the optimizer, number of epochs, configuration of regularizations, and nets in order to arrive to the optimal combination that shows the best accuracy on validation and then with the chosen optimal combination make a test and check the test accuracy.


## Models


For our exercise we have tested two net configurations:
- Siamese Decision Network with simple loss. traditional Siamese net where two nets are calculated in parallel, the results are concatenated, and the resulting loss is used to backpropagate
- Siamese Decision Network with the average of two losses. A different approach in which first we feed the Siamese net with the positive (true same person pairs) cases and we obtain its loss, then we feed again the Siamese net with negative cases (non same person pairs). Then the average of both losses is calculated and used to backpropagate. In this way we ensure that exactly the 50% of the cases are of one case and 50% of the opposite, this mechanism helps the net to train better in the opinion of the author of this project.


![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/image1%20-%20siamese.png?raw=true)

*Image  from Amazon [Amazon](https://aws.amazon.com/es/blogs/machine-learning/combining-deep-learning-networks-gan-and-siamese-to-generate-high-quality-life-like-images/)

For Each branch we choose two Pretrained Nets:
- VGG
- Alexnet

Detail of the net structures:

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


We use only two colab codes:

-One for the Siamesse Decision Network with single loss: "ARTUR PALOMINO CODE1.ipynb"

-One for the Siamesse Decision Network with two losses: "ARTUR PALOMINO CODE2.ipynb"


## Code  "ARTUR PALOMINO CODE1.ipynb" Structure

This code is for testing different configurations for a VGG with single loss.

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





## Code  "ARTUR PALOMINO CODE2.ipynb" Structure

This code is for testing different configurations for a VGG and an Alexnet with two losses.

We split the code in the following sections:

MOUNTING DRIVE

↳ we mount google drive CFPW images that were previously uploaded 

IMPORTING PACKAGES

↳ we import different packages needed for the execution

CREATING DOWNLOADER CLASS

↳ The downloader class is used to feed the net with CFPW images

CREATING MODELS

↳ We have a VGG decision network and Alexnet with two losses.

TRAIN

↳ training function

TEST

↳ test function

MAIN

↳ main function

DOWNLOAD FILES (JUST IN CASE)

↳ download files from original websit in case needed

After this different tests are calculated with different parameter configurations


## Results for Code  "ARTUR PALOMINO CODE2.ipynb"

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image18%20-%20results%20code2.PNG?raw=true)

#### TEST A: ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|F|DECISION NET 2 LOSS|81|80|T|15|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image19%20-%20testA%20loss.png?raw=true)]()

##### Conclussions:
The configuration is not optimal as it shows overfit. The accuracy is nice and we can see the the optimal accuracy is achieved fast, in less than 10 epochs. Another interesting thing is that this net trains 60 epochs in les than 60 minutes while VGG lasts 4 times more.


#### TEST B: ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|T|DECISION NET 2 LOSS|84|82|F|59|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image20%20-%20testA%20loss.png?raw=true)]()

##### Conclussions:
The configuration doesn't show overfit moreover we arrive to an outstanding validation accuracy of 84 and a test accuracy of 82, this is by far the best combination until that moment. This is in fact one of the best options for Alexnet in this document.


#### TEST C: VGG, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|F|F|F|DECISION NET 2 LOSS|80|78|F|31|

##### Loss image & Accuracy image:
[![N|Solid](https://raw.githubusercontent.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/master/images/Image21%20-%20testC%20loss.png?raw=true)]()

##### Conclussions:
The configuration using a VGG shows worst results than the Alexnet in the same conditions, without data augmentation, without pretraining and without freeze, by the other hand the model has overfit.

#### TEST D: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|F|F|DECISION NET 2 LOSS|84|83|T|5|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image22%20-%20testD%20loss.png?raw=true)]()

##### Conclussions:
The configuration using a VGG and pretrain we obtain a nice validation accuracy of 84 and a test accuracy of 83, better than the last model except for the fact that this one has overfit.


#### TEST E: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|F|T|DECISION NET 2 LOSS|86|84|T|15|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image23%20-%20testE%20loss.png?raw=true)]()

##### Conclussions:
The configuration shows the best validation accuracy of the whole set of combinations, an 86%. Still have some overfit but validation curve increases gradually. It is probably the best option for the VGG with two losses.


#### TEST F: VGG, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

(No image)

##### Conclussions:
In order to test the learning rate 1e-3 we run again our last model, the results are the worst until the moment, the problem is that the accuracy never improves from 50%, the reason is that it's not possible for the model to arrive to the minimum loss beacuse the interval at every step is too big and the minimum is allways in the middle of the last two steps.




#### TEST G: VGG, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5E-4|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image25%20-%20testG%20loss.png?raw=true)]

##### Conclussions:
In order to test again the model, the learning rate is fixed at 5e-4. We find exactly the same scenario as in testG.



#### TEST H: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|T|T|DECISION NET 2 LOSS|84|84|T|16|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image26%20-%20testH%20loss.png?raw=true)]

##### Conclussions:
In this combination we include the option freeze, freezing 7 convolutions of the pretrained layers on the VGG. The test accuracy shows the best results of the table, the problem again is the overfit


#### TEST I: ALEXNET, LR 5E-4 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|T|DECISION NET 2 LOSS|83|81|F|239|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image27%20-%20testI%20loss.png?raw=true)]

##### Conclussions:
In this combination we run more epochs for the alexnet of the Test B. As we can see, the combination still don't have overfit, is quite robust. The only problem is that we don't improve the accuracy level of Test B.




#### TEST J: ALEXNET, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|5E-4|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image28%20-%20testJ%20loss.png?raw=true)]

##### Conclussions:
In this combination similarly to the case of test F and G we don't have convergency to the minimum of the loss




#### TEST K: ALEXNET, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-3|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image29%20-%20testk%20loss.png?raw=true)

##### Conclussions:
In this combination similarly to the case of test F and G we don't have convergency to the minimum of the loss


#### TEST L: ALEXNET, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|T|F|T|DECISION NET 2 LOSS|83|82|T|137|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image30%20-%20testL%20loss.png?raw=true)

##### Conclussions:
In this combination we have a nice accuracy of test and validation data but still has overfit


#### TEST M: ALEXNET, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretrain | Freeze | Data Augmentation | Variations | Best epoch val.accur. | Best e.Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|T|T|T|DECISION NET 2 LOSS|78|78|T|179|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image31%20-%20testM%20loss.png?raw=true)

##### Conclussions:
In this combination, freezing 3 layers of the convolutions we have poor accuracies and still shows overfit

## GENERAL CONCLUSSIONS

VGG achieves better validation accuracy but generally with overfit.

>Best option with VGG: 2 losses decision network, Pretrained, Adam, lr:1e-5, >wd=0, with data augmentation, dropout, 86 val Accuracy, 82 Test accuracy at >epoch 15

 
Alexnet achieves good validation accuracy normally without overfit.
Freeze seems to not work so well, although it makes the net train faster.
Data augmentation makes the difference, there are increases of performance close to 4%. Not all transformations work well
Alexnet can train in 1 hour 60 epochs while VGG needs 4 hours for the same number of epochs
LR of 1e-3 normally is not a good option and 1e-5 & 5e-4 normally worked better


## DEMO

In order to check the results in a real example we add a piece of code at the testing funcion where two pairs of images are shown in real time. The row of the output corresponds allways to different persons, the second pair correpond to the same person. The model tryes to predict whether te person is the same or not. So if in the first row it predicts "Not the same person" this is correct, if in the second row ir predicts "The same person" then it is correct too.

##### Example with TEST E: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0 

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/DEMO.PNG?raw=true)


