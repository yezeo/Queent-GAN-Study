# LSGAN review

## 0. Abstract
vanilia GAN의 D는 sigmoid cross entropy loss function을 사용함
이는 vanishing gradient 문제가 있어 Least Squares GAN을 제안함으로써 이 문제를 극복함
LSGAN의 장점은 GAN 보다 high quality의 이미지 생성, training 과정에서 안정화에 도움.

## 1. Intro
![6_1](https://user-images.githubusercontent.com/66044830/140878575-31a15573-3c9c-4087-890d-291ed1be8ffa.JPG)

(b) : fake sample이 G를 업데이트 하기위해 사용될 때 D는 진짜 데이터라고 믿는 상황
하지만 D를 잘 속였지만 실제 데이터 분포와는 다른 것을 알 수 있다. 즉, 분홍색 point를 실제 데이터 분포로 당겨와야 high quality image가 만들어 질 것임
이 때 D에 least squares loss function이 사용됨. 왜냐하면 최소제곱법을 함으로써 멀리 떨어져있는 데이터에 대해 패널피를 주기 때문이다.

(c) : 패널티를 받은 fake sample들이 실제 데이터 분포로 끌어 당겨지고 있음을 확인할 수 있음.

-학습의 안정성
loss function을 수정함으로써 GAN의 vanishing gradients 문제를 해결함.

## 3. Method
![6_2](https://user-images.githubusercontent.com/66044830/140878567-4b528976-b805-474c-9dd3-e64a985c353d.JPG)
a : data, b : fake data label, c : G가 가짜데이터에 대해 D가 믿도록 원하는 value
a,b,c를 결정하는 방법 : minimizing the Pearson X^2 divergence
=> b-c =1 & b-a =2

## 4. Result
![사진3](https://user-images.githubusercontent.com/66044830/140878574-d211d305-87cf-45db-b34b-8048bb8e8865.JPG)
위의 사진을 보면 GAN으로 학습한 것 보다 LSGAN으로 생성한 이미지가 학습과정에서 훨씬 더 안정적인 모습을 띈다는 것을 확인할 수 있다.