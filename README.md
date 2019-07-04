# AIDL-PROJECT-ARTURO-PALOMINO
ARTIFICIAL INTELLIGENCE AND DEEP LEARNING PROJECT - UPC 

# Index
[1. Introduction](#introduction)

[2. Codes Structure](#codes_structure) 

[3. Motivation](#motivation)

[4. Models](#models)

[5. Parameters](#parameters)

[6. Results](#results) 

[8.General conclusions](#general_conclusions)

[9.Future work](#future_work)

[10.Demo](#demo)

[11.Instructions for Demo](#instructions_for_demo)

[References](#references)

## Introduction
Student: Arturo Palomino
Team: 4
Results of the final project of the Deep Learning and Artificial Intelligence post degree of the UPC talent-school.

For this project different architectures have been tested in order to train a model able to recognize if two faces correspond to the same person. For this purpose, I use Siamese networks that returns a classification output with a binary answer, in my case the answer is a vector of two elements or it can be one element with a dichotomic variable, depending on how the last layer of the net is configured, in this exercise I use only the first approach. If the first element of the vector is higher than the second element then I can say that the person is the same, in other cases the person is not the same.

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/image1%20-%20siamese.png?raw=true)
Image1: siamese network
*Image  from Amazon [Amazon](https://aws.amazon.com/es/blogs/machine-learning/combining-deep-learning-networks-gan-and-siamese-to-generate-high-quality-life-like-images/)

My first approach is to test different parameters of the optimizer, number of epochs, configuration of regularizations and nets, in order to arrive to the optimal combination that shows the best accuracy on validation and then with the chosen optimal combination make a test and check the test accuracy. In my case the best accuracy we obtained arrived to 86% and the configuration corresponds to a VGG with adam optimizer, with a learning rate of 1e-5 with weight decay of 0, pretrained but without freezing the layers and with data augmentation, for the data augmentation I used RandomHorizontalFlip, RandomAffine.

## Codes Structure
```bash
AIDL-PROJECT-ARTURO-PALOMINO
├── images
│   ├── *.png
├── DEMO FILES AND WEIGHTS
│   ├── Artur Demo.ipynb
│   ├── links to the weights and needed files.txt
│   ├── requirements.txt
├── ARTUR PALOMINO CODE1.ipynb
├── ARTUR PALOMINO CODE2.ipynb
└── README.md
```

I mainly use two colab codes that the reader can check and see the different results already executed step by step. The two colab codes are related to the two architectures I used:

-One for the Siamesse Decision Network with single loss: "ARTUR PALOMINO CODE1.ipynb"

-One for the Siamesse Decision Network with two losses: "ARTUR PALOMINO CODE2.ipynb"


### CODE  "ARTUR PALOMINO CODE1.ipynb" STRUCTURE

This code is for testing different configurations for a VGG with single loss.

I split the code in 7 sections:


##### ARTUR PALOMINO CODE1.ipynb
├─  MOUNTING DRIVE   -> I mount google drive CFPW images that were previously uploaded 

├─  IMPORTING PACKAGES   -> I import different packages needed for the execution

├─  CREATING DOWNLOADER CLASS  -> The downloader class is used to feed the net with CFPW images

├─  CREATING SIAMESE NETWORKS(VGG...) -> I have a VGG decision network and VGG linear network (that last not used)

├─ UTILITIES FOR METRICS OF OUR MODELS  -> With this section I obtain the validation and training accuracy and losseS

├─ LOOP FOR TRAINING  -> In this section I create the model, calculate the loss and backpropagate

├─ LOOP FOR VALIDATION  ->In this section the model is calculated with validation images


After this, different test are calculated with different parameter configurations, the reader can read in the colab the last execution done by me and the results.


├─ LTEST 1: LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION

├─ LTEST 2: LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITH DATA AUGMENTATION

├─ LTEST 3: LR=5E-4, WD=0, N_EPOCHS=60 SIAMESE DECISION WITH DATA AUGMENTATION

├─ LTEST 4: LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

├─ LTEST 5: LR=5E-4, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

├─ LTEST 6: LR=1E-5, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE



### CODE  "ARTUR PALOMINO CODE2.ipynb" STRUCTURE

This code is for testing different configurations for a VGG and an Alexnet with two losses.

I split the code in the following sections:

├─ MOUNTING DRIVE -> I mount google drive CFPW images that were previously uploaded 

├─ IMPORTING PACKAGES ->  I import different packages needed for the execution

├─ CREATING DOWNLOADER CLASS -> The downloader class is used to feed the net with CFPW images

├─ CREATING MODELS ->  I have a VGG decision network and Alexnet with two losses.

├─ TRAIN ->  Training function

├─ TEST -> Test function

├─ MAIN -> Main function

├─ DOWNLOAD FILES (JUST IN CASE) -> Download files from original website in case needed


After this, different tests are calculated with different parameter configurations. The reader can check the last executions and the results obtained by me step by step:


├─ TEST A: ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

├─ TEST B: ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

├─ TEST C: VGG, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

├─ TEST D: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=0 FREEZE=0

├─ TEST E: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

├─ TEST F: VGG, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

├─ TEST G: VGG, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

├─ TEST H: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1

├─ TEST I: ALEXNET, LR 5E-4 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

├─ TEST J: ALEXNET, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

├─ TEST K: ALEXNET, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

├─ TEST L: ALEXNET, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

├─ TEST M: ALEXNET, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1




## Motivation

The main motivation for this project a part from learning how to improve a model was to be able to develop an algorithm able to detect persons that walk in front of a webcam, as an enthusiast of drones and in general to aeromodels I always wanted to know if with the image sent from my drone to my movile I would be able to point to the face of a person and detect if that person is someone I know or it's a stranger. Of course another big objective but a little bit more difficult is to be able to develop an application for automatic piloting system that empowers the machine to make short itineraries without human intervention. At that sense recognizing simple objects is the first step of a long road.

## Models


For my exercise I have tested two net configurations:
- Siamese Decision Network with simple loss. traditional Siamese net where two nets are calculated in parallel, the results are concatenated, and the resulting loss is used to backpropagate.

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/ImageA%20-%20structure1.PNG?raw=true)
Image2: Siamese network with 1 loss

- Siamese Decision Network with the average of two losses. A different approach in which first I feed the Siamese net with the positive (true same person pairs) cases and I obtain its loss, then I feed again the Siamese net with negative cases (non same person pairs). Then the average of both losses is calculated and used to backpropagate. In this way I ensure that exactly the 50% of the cases are of one case and 50% of the opposite, this mechanism helps the net to train better in the opinion of the author of this project.

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/ImageB%20-%20structure2.PNG?raw=true)
Image3: Siamese network with 2 losses



For Each branch I choose two Pretrained Nets:
- VGG [3]
- Alexnet [4]

Detail of the net structures:

- VGG :

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image2%20-%20vgg16%20b.png?raw=true)

- ALEXNET

*Image  from Neurohive [Neurohive](https://neurohive.io/en/popular-networks/vgg16/)
![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image3%20-%20AlexNet-CNN-architecture-layers.png?raw=true)

*Image  from Researchgate [Researchgate](https://www.researchgate.net/figure/AlexNet-CNN-architecture-layers_fig1_318168077)



## Parameters


For the different tests I use the following parameters



![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image4%20-%20parameters.PNG?raw=true)
For the Net in general all the members of the group made a test with a VGG, but I have also made other tests with other CNN like Alexnet.

The optimizer I have used is Adam, this option is faster but in the result conclusions I will explain the benefits of considering other options.

For the Adam I consider in general 2 main learning rates 1e-3, 5e-4 but I will see that this kind of data needs in some cases lower values.

We don't consider changing Weight decay, at initial stages of the project I decided to avoid changing it to 1 because of anomalous results.

We consider combinate dropout in some cases to avoid overfit.

For the pretrained parameter, I use two options, fix the pretrained weights with Imagenet [1], then finetune letting the net to learn all the weights, and another exercise where I fix the pretrained weights and then I freeze part of the convolutional layers (7 layers in the VGG, 3 in the Alexnet).

Then I use two different architectures explained in the previous point.


## RESULTS

## CODE  "ARTUR PALOMINO CODE1.ipynb" EXECUTIONS

The following table shows the different results for the second architecture for the siamese network with two losses.
![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image5%20-%20results%20code1.PNG?raw=true)

### TEST 1

#### LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|F|F|DECISION NET|77|76|T|54|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image6%20-%20test1%20loos.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image7%20-%20test1%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration presents overfit, I can see a separation between validation and training loses and a convexity at 30 epochs

### TEST 2

#### LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITH DATA AUGMENTATION

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|F|T|DECISION NET|79|80|F|58|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image8%20-%20test2%20loos.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image9%20-%20test2%20accuracy.png?raw=true)]()

##### Conclussions:
The configuration now doesn't present overfit, I have  a decent Validation accuracy and a nice test accuracy of 80%. As I will see in further combinations this one is the best candidate for a VGG with simple loss.

### TEST 3

#### LR=5E-4, WD=0, N_EPOCHS=60 SIAMESE DECISION WITH DATA AUGMENTATION

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5E-4|0|T|16|T|F|T|DECISION NET|80|78|F|47|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image10%20-%20test3%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image11%20-%20test3%20accuracy.png?raw=true)]()

##### Conclusions:
The configuration presents a nice validation accuracy and no overfit, by the other hand I can see that the accuracy is lower than previous test for testing sample, that lead us to prefer test2 for being more stable in that sense.

### TEST 4


#### LR=1E-3, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|T|16|T|T|T|DECISION NET|50|50|...|0|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image12%20-%20test4%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image13%20-%20test4%20accuracy.png?raw=true)]()

##### Conclusions:
In this execution I add freeze to the first 7 convolutional layers. The configuration presents strange values for losses and accuracies this is probably due to the fact that the learning rate is too big and the minimum loss is at a certain point that is inaccessible with an interval of 1e-3. I try in the next test to decrease this value to 5e-4 in order to verify my hypothesis.


### TEST 5:

#### LR=5E-4, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5e-4|0|T|16|T|T|T|DECISION NET|81|79|T|56|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image14%20-%20test5%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image15%20-%20test5%20accuracy.png?raw=true)]()

##### Conclusions:
The configuration with this learning rate solves my previous test problem and now I arrive to the minimum loss easily. The bad point is that I continue having overfit.


### TEST 6:


#### LR=1E-5, WD=0, N_EPOCHS=60 SIAMESE DECISION WITHOUT DATA AUGMENTATION WITH FREEZE

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5e-4|0|T|16|T|T|T|DECISION NET|83|83|T|55|

##### Loss image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image16%20-%20test6%20loss.png?raw=true)]()

##### Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image17%20-%20test6%20accuracy.png?raw=true)]()

##### Conclusions:
The configuration with the freeze and the learning rate of 1e-5 works great, I get a validation accuracy of 83 and a testing accuracy of 83 too. The problem that persists is the overfit that have been reflected in all tests with freeze that I have done.



## CODE  "ARTUR PALOMINO CODE2.ipynb" EXECUTIONS

The following table shows the different results for the first architecture, the siamese network with one loss.

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image18%20-%20results%20code2.PNG?raw=true)

### TEST A:

#### ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|F|DECISION NET 2 LOSS|81|80|T|15|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image19%20-%20testA%20loss.png?raw=true)]()

##### Conclussions:
The configuration is not optimal as it shows overfit. The accuracy is nice and I can see the optimal accuracy is achieved fast, in less than 10 epochs. Another interesting thing is that this net trains 60 epochs in less than 60 minutes while VGG lasts 4 times more.


### TEST B:

#### ALEXNET, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|T|DECISION NET 2 LOSS|84|82|F|59|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image20%20-%20testA%20loss.png?raw=true)]()

##### Conclusions:
The configuration doesn't show overfit moreover I arrive to an outstanding validation accuracy of 84 and a test accuracy of 82, this is by far the best combination until that moment. This is in fact one of the best options for Alexnet in this document.

### TEST C:

#### VGG, LR 1E-5 WD0 PRETRAIN=0 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|F|F|F|DECISION NET 2 LOSS|80|78|F|31|

##### Loss image & Accuracy image:
[![N|Solid](https://raw.githubusercontent.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/master/images/Image21%20-%20testC%20loss.png?raw=true)]()

##### Conclusions:
The configuration using a VGG shows worst results than the Alexnet in the same conditions, without data augmentation, without pretraining and without freeze, by the other hand the model has overfit.


### TEST D:

#### VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=0 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|F|F|DECISION NET 2 LOSS|84|83|T|5|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image22%20-%20testD%20loss.png?raw=true)]()

##### Conclusions:
The configuration using a VGG and pretrain I obtain a nice validation accuracy of 84 and a test accuracy of 83, better than the last model except for the fact that this one has overfit.

### TEST E:

#### VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|F|T|DECISION NET 2 LOSS|86|84|T|15|

##### Loss image & Accuracy image:
[![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image23%20-%20testE%20loss.png?raw=true)]()

##### Conclusions:
The configuration shows the best validation accuracy of the whole set of combinations, an 86%. Still have some overfit but validation curve increases gradually. It is probably the best option for the VGG with two losses.


### TEST F:

#### VGG, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-3|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

(No image)

##### Conclusions:
In order to test the learning rate 1e-3 I run again my last model, the results are the worst until the moment, the problem is that the accuracy never improves from 50%, the reason is that it's not possible for the model to arrive to the minimum loss because the interval at every step is too big and the minimum is always in the middle of the last two steps.


### TEST G:

#### VGG, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|5E-4|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image25%20-%20testG%20loss.png?raw=true)]

##### Conclusions:
In order to test again the model, the learning rate is fixed at 5e-4. I find exactly the same scenario as in testG.

### TEST H:

#### VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|VGG|ADAM|1E-5|0|F|16|T|T|T|DECISION NET 2 LOSS|84|84|T|16|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image26%20-%20testH%20loss.png?raw=true)]

##### Conclusions:
In this combination I include the option freeze, freezing 7 convolutions of the pretrained layers on the VGG. The test accuracy shows the best results of the table, the problem again is the overfit

### TEST I:

#### ALEXNET, LR 5E-4 WD0 PRETRAIN=0 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|F|F|T|DECISION NET 2 LOSS|83|81|F|239|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image27%20-%20testI%20loss.png?raw=true)]

##### Conclusions:
In this combination I run more epochs for the alexnet of the Test B. As I can see, the combination still don't have overfit, is quite robust. The only problem is that I don't improve the accuracy level of Test B.


### TEST J:

#### ALEXNET, LR 5E-4 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|5E-4|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image28%20-%20testJ%20loss.png?raw=true)]

##### Conclusions:
In this combination similarly to the case of test F and G I don't have convergence to the minimum of the loss


### TEST K:

#### ALEXNET, LR 1E-3 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-3|0|F|16|T|F|T|DECISION NET 2 LOSS|50|50|...|0|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image29%20-%20testk%20loss.png?raw=true)

##### Conclussions:
In this combination similarly to the case of test F and G I don't have convergency to the minimum of the loss

### TEST L:

#### ALEXNET, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|T|F|T|DECISION NET 2 LOSS|83|82|T|137|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image30%20-%20testL%20loss.png?raw=true)

##### Conclussions:
In this combination I have a nice accuracy of test and validation data but still has overfit

### TEST M:

#### ALEXNET, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=1

|  Net | Optim  | LR | WD | Drop out  | Batch sz  |  Pretr | Freeze | Data Augm | Architec. | Val.Acc | Test accur. | Overfit | B.epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Alexnet|ADAM|1E-5|0|F|16|T|T|T|DECISION NET 2 LOSS|78|78|T|179|

##### Loss image & Accuracy image:

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/Image31%20-%20testM%20loss.png?raw=true)

##### Conclussions:
In this combination, freezing 3 layers of the convolutions I have poor accuracies and still shows overfit

## General conclusions

VGG achieves better validation accuracy but generally with overfit.

>Best option with VGG: 2 losses decision network, Pretrained, Adam, lr:1e-5, >wd=0, with data augmentation, dropout, 86 val Accuracy, 82 Test accuracy at >epoch 15

 
Alexnet achieves good validation accuracy normally without overfit.
Freeze seems to not work so well, although it makes the net train faster.
Data augmentation makes the difference, there are increases of performance close to 4%. Not all transformations work well
Alexnet can train in 1 hour 60 epochs while VGG needs 4 hours for the same number of epochs
LR of 1e-3 normally is not a good option and 1e-5 & 5e-4 normally worked better


## Future work

In future works I will try to run other networkds (resnet, densenet, googlenet) and losses (triple loss, constractive loss). Another improve according with last class (june 27th) recomendations to arrive better to global minimum I will try to rerun the best combinations with pure SGD, as it seems to be slower but with better results.

## DEMO

In order to check the results in a real example I add a piece of code at the testing function where two pairs of images are shown in real time. The first row of the output corresponds always to different persons, the second pair correspond always to the same person. The model tries to predict whether the person is the same or not. So if in the first row it predicts "Not the same person" this is correct, if in the second row ir predicts "The same person" then it is correct too.

##### Example with TEST E: VGG, LR 1E-5 WD0 PRETRAIN=1 DATA AUGM=1 FREEZE=0 

![N|Solid](https://github.com/bemolicida/AIDL-PROJECT-ARTURO-PALOMINO/blob/master/images/DEMO.PNG?raw=true)

## Instructions for Demo

In order to execute the demo the reader will need to install the packages in the requirement file found inside the demo path with the following instruction:
```
Pip install -r requirements.txt
```
This will configure python to execute the “Artur Demo.ipynb” code that the reader can find in the “DEMO FILES AND WEIGHTS” folder. 
Previously the reader will need to download the weights of the test B configuration of the Results chapter. In the same folder the reader will find a file named “links to the weights and needed files.txt” where will find everything: 
-	Artur_weights_siamese_epoch 59__modelAlexnet_amp 1_pretr 0_lr 1e-05_wd 0.pt Artur_Pair_list_F.txt
-	Artur_Pair_list_P.txt
-	Artur_test.py
All the files must be placed in a subfolder named “Weights_here” following this structure:

```bash
AIDL-PROJECT-ARTURO-PALOMINO 
├── Artur Demo.ipynb
│   ├── WEIGHTS_HERE
│   ───├── Artur_weights_siamese_epoch 59__modelAlexnet_amp 1_pretr 0_lr 1e-05_wd 0.pt
│   ───├── Pair_list_F
│   ───├── Pair_list_P
│   ───├── Artur_test.pt
```
The reader will need to change the paths at the second section of the code “Packages and other variables”. This is an example of how it’s in my current configuration:
```
rutaDataColab="gdrive/My Drive/2019_AIDL_TEAM4/colab_face_detection_siamesa/Presentacio/Demo/Weights_here/"
rutaCodeColab="gdrive/My Drive/2019_AIDL_TEAM4/colab_face_detection_siamesa/Presentacio/Demo/"
rutaWeightsColab="gdrive/My Drive/2019_AIDL_TEAM4/colab_face_detection_siamesa/Presentacio/Demo/Weights_here/"
```

The it is necessary to run sequentially every section in order until the end.

Please don’t hesitate to contact me directly to my email if there is any doubt: estadisticman@hotmail.com

## References

[1] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). Imagenet large scale visual recognition challenge. arXiv preprint arXiv:1409.0575.

[2] Chopra, S., Hadsell, R., & LeCun, Y. (2005, June). Learning a similarity metric discriminatively, with application to face verification. In CVPR (1) (pp. 539-546).

[3] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." ICLR 2015. 

[4] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks.



