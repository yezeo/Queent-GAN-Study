# Generative Adversarial Nets review

## 0. Abstract
ACGAN : class에 따라 이미지를 합성함(Discriminator가 real/fake를 판별하면서 class prediction도 같이 학습함)

## 1. Intro
![5-1](https://user-images.githubusercontent.com/66044830/136880523-17c769b0-c73f-4b67-b987-6c68c49c5498.JPG)
![5-2](https://user-images.githubusercontent.com/66044830/136880525-3f78bcc6-1bf5-41fb-9b00-be4ae31bae14.JPG)

## 2. AC-GANs(Auxiliary Classifier GAN)
* **Generator**는 noise를 sampling할 때 class label도 같이 sampling한다.

$X_{fake} = G(c,x)$

* **Discriminator** : source와 class label의 확률분포를 바탕으로 학습 함.

$P(S|X), P(C|X) = D(X)$

**Object Function**
![5-4](https://user-images.githubusercontent.com/66044830/136880591-24fda3a0-99d3-41d5-8e22-d6a01354e085.JPG)

(2) 식은 log likelihood of the correct source이고, (3) 식은 log likelihood of the correct class이다.

=> Discriminator는 L_s + L_c를, G는 L_s - L_c를 최대화하는 방향으로 학습함

real/fake 쪽을 판별하는 부분은 G와 D가 adversarial 학습을 하지만, class prediction은 adversarial하지 않게 학습함.

또한, 이전 연구들은 class 수를 smfflaus quality가 줄어들었지만, ACGAN은 class별로 큰 데이터를 나눈 후 subset에 대해 G,D를 학습할 수 있음 -> 이 결과로 안정적이며 질 좋은 output을 얻을 수 있음

## 3. Result
![5-5](https://user-images.githubusercontent.com/66044830/136880520-1444f260-e1d6-4f7b-acc3-d1160db297e2.JPG)

ACGAN은 64x64, 128x128의 resolution을 가진 image 생성. 고해상도의 이미지는 저해상도의 sample을 단순히 resizing하는 것이 아니라 실제 이미지의 특성을 더 잘 반영함을 확인 가능함.

**Measuring the Diversity of Generated Images**
- image discriminability를 측정하기 위한 metric(= MS-SSIM metric)제안
  
각 class 내의 이미지 수가 적다면 생성모델은 해당 class 내의 사진들을 암기할 수도 있음. mode collapse가 생길 수도 있는데, Inception score로는 측정 불가.

=> 생성모델을 평가하려면 다양성을 측정하는게 중요. 이를 위해 MS-SSIM의 method를 사용하여 모델의 다양성 평가(다양성이 높을수록 MS-SSIM score이 낮게 나옴)

        ACGAN에서는 vanilla GAN과 다르게 고해상도의 이미지를 생성할 수록 다양한 이미지가 생성됨(= mode collapse에 빠질 확률이 줄어듦)