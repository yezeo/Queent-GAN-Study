# Conditional Generative Adversarial Nets review

## 0. Abstract
CGAN : the conditional version of GAN constructed by feeding the data y, and contition on to both the G and D

- this model can generate MNIST digits conditioned on class labels

-----------
## 1. Intro
Adversarial net은 Markov chain이 전혀 필요하지 않고, backpropagation만 사용하여 기울기를 얻을 수 있다는 장점이 있다. 그리고 학습 중에 추론이 필요하지 않으며 다른 요소들을 모델에 쉽게 통합할 수 있음

하지만 vanilla GAN에서는 generated data에 대한 control이 불가능함. 즉, MNIST에서 '1'을 생성하고 싶다고 1만을 목표로 생성할 수 없다는 것(생성하고자 하는 데이터셋을 지정할 수 없음)

👍이에 반해 CGAN은 possible to direct the data generation process. ➡ 이걸 conditioning이라 하고 class label / 다른 양식의 data를 기반으로 한다.

----
## 2. Related Works
### 2.1 Multi-modal Learning For Image Labeling

despite the successes of Supervised NN, it's hard to predict the extremly large number of data. 

그리고 input : output = 1:1 mapping에 focus를 두고 학습한 경우가 많았는데 이제는 1:多 mapping으로 이어짐. 즉, 이미지에는 여러 태그가 존재할 수 있고, 인간은 그에 대해 다양한 단어를 사용할 수 있음 => 이는 conditional probabilistic generative model로 해결 가능

------------
## 3. Conditional Adversarial Nets
### 3.2 CGAN
GAN과의 차이점 : y라는 새로운 정보 추가
즉, MNIST로 예를 들자면 4를 생성하고 싶을 땐 [0 0 0 0 1 0 0 0 0 0]을 y로 같이 넣어줌

![3_1](https://user-images.githubusercontent.com/66044830/135027951-9ee82644-50a3-45c1-88e9-75da2895605f.JPG)

![3-2](https://user-images.githubusercontent.com/66044830/135027957-e6f94a26-e1e7-4dd7-8c14-fa3a96efec91.JPG)

기존의 GAN과 모양은 비슷하지만 G와 D에 조건 y가 추가됨

------
## 4. Experimental Results
### 4.1 Unimodel
G, D 모두 conditioning을 추가함. 
MNIST에서는 생성하고 싶은 수에 대한 one-hot을 y로 집어 넣어준다.

### 4.2 Multi-modal
word vector와 같은 다른 약식의 데이터를 input
->pix2pix, cyclegan과 같은 형태로 발전

------
## 5. Future work
동시에 여러 태그를 사용하는 방향으로 발전 가능