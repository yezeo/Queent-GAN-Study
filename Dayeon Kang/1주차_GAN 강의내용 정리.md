# GAN 스터디 

## **1. Probability Distribution**
![1](https://user-images.githubusercontent.com/66044830/133193834-7b117052-8694-49f2-96bf-42a4e48a9839.JPG)
자료의 출현 빈도를 확률을 명시하는 식으로 train data의 prob dist를 만들어나감.

그리고 x4 벡터에 해당하는 이미지는 임의로 만들어낸 괴상한 이미지인데 이는 train dataset에는 없는 이미지 이기 때문에 prob는 0에 근사한다

✨그럼 여기서 Generative model이 하는 역할은?
![2](https://user-images.githubusercontent.com/66044830/133193837-4d0ff007-5e45-4ade-b4f7-515c40cf25a4.JPG)

train dataset의 prob distribution과 유사한 분포를 만들어나가는 것. 빨간색은 모델이 생성한 이미지의 확률 분포. 그래서 두개의 분포 차이를 줄여주는게 G의 목표가 된다

## **2. Generative Adversarial Networks**
![3](https://user-images.githubusercontent.com/66044830/133193838-92f0589b-f2d6-4c94-b5b9-cda404704951.JPG)

[train 과정]

1. D를 먼저 train 시킴.
   - 진짜와 가짜를 구별하기 위해
   - input size는 64x64x3이고 output size는 1 (sigmoid를 거쳐 0.5를 기준으로 자름)

2. G는 임의의 코드를 받아서 이미지를 생성해내야한다
   - D를 속여야하기 때문에 output이 1이 나오도록 학습해야한다

[Object function]
![4](https://user-images.githubusercontent.com/66044830/133193839-070a5809-70e3-476d-b495-9a331f872fc2.JPG)
0< D(x) < 1 이어야 한다

- z가 G의 입력값으로 들어감
   - 가우시안 분포 / 유니폼 분포에서 100차원의 sampling을 한 random한 vector를 G한테 줌 -> G가 가짜 이미지를 생성해냄 -> 그럼 D는 0에 가까운 값을 내놔야함 

=> 이게 $log(1-D(G(Z)))$

그래서 이 loss func으로 D가 이를 최대화하도록 학습을 하면 진짜 이미지는 1에 가깝도록 학습을 하고 가짜 이미지는 0에 가깝도록 학습을 한다

### Code
![5](https://user-images.githubusercontent.com/66044830/133193840-72ed6a9a-ebe6-4301-8f71-2e2a0050b3dc.JPG)

- G의 아웃풋은 이미지여야 하기 때문에 output size는 784

![6](https://user-images.githubusercontent.com/66044830/133193841-9fd3fc32-1de3-46e7-af58-02bb07c2367e.JPG)



## **3. Variants of GAN**
### 1) DCGAN
![7](https://user-images.githubusercontent.com/66044830/133193843-83d0c7cb-df33-425c-a9f1-91ccd41c8506.JPG)
CNN을 사용해서 D를 구현을 하고 deconvolutional nn(혹은 transpose convolution)을 통해서 G를 만든 모델

DCGAN의 핵심 : Pooling layer를 사용하지 않았다는 것. 그래서 stride size가 2 이상인 convolution / deconvolutiond을 사용

### 2) LSGAN
![8](https://user-images.githubusercontent.com/66044830/133193846-0321051e-5c2d-463f-aca9-1733ea93944d.JPG)

idea는 D에서 sigmoid를 없앴고 cross entropy loss를 L2 loss로 바꿔준 것

### 3) SGAN
![9](https://user-images.githubusercontent.com/66044830/133193847-c9fd43d2-60f7-409b-b058-ef0d77afe76a.JPG)
D가 진짜 가짜를 구분하지 않고 class를 구분하게 됨

그래서 기존 GAN이랑 다른 건 어떤 걸 생성할 지 input 값으로 class를 주면 해당 이미지를 생성하는 것.

#### (1) ACGAN
![10](https://user-images.githubusercontent.com/66044830/133193848-c62be387-78f8-4fe5-9a64-eb0f5cc63446.JPG)


## **4. Extensions of GAN**
### 1) CycleGAN : Unpaired Image to Image Translation
![11](https://user-images.githubusercontent.com/66044830/133193851-fbfa19f1-6b2c-4f43-b85f-9b878da7d6bf.JPG)
모양은 유지가 되는 걸로!

### 3) StackGAN : Text to Photo realistic Image Synthesis
![12](https://user-images.githubusercontent.com/66044830/133193852-e2204880-2175-4f61-97c0-c8858a07f1e3.JPG)
