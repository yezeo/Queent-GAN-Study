# DCGAN ; Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks 

## 0. Abstract
- CNN을 이용한 unsupervised learning

## 1. Intro
- convolution GAN architecture에 대한 제약 조건 제안, 평가 -> stable한 training
- 이미지 분류 문제를 위한 train된 판별자 사용
- GAN에 의해 학습된 filter를 시각화하고 특정한 filter들이 특정 객체들을 학습하는 것을 show
  
## 2. Related work
### 2.1 Representation learning from unlabeled data
이미지 -> 파워풀한 이미지 표현을 train하기 위해 이미지 patch의 계층적 clustering 가능

### 2.2 Generating Natural Images
Non-parametric 모델 : 종종 존재하는 이미지들의 db로부터 되고, image들의 path르 매칭하는 것은 texture synthesis, super-resolution, in-painting과 같은 분야에 사용되어짐

Parametric : 이미지들을 생성하기 위한 variational sampling 접근법은 약간의 성공을 이루었지만 샘플들은 blur가 되는 현상을 겪는다.

### 2.3 Visualizing the internals of CNNs
NN을 사용하는 것에 대한 지속적인 비판은 알고리즘 형태에서 네트워크가 어떻게 동작하는 지에대한 이해가 부족한 블랙박스 방법이기 때문.

## 3. Approch and Model Architecture
CNN architecture을 적용하고 3가지를 변경함

        1. 결정적인 spatial pooling 함수를 strided convolution으로 대체. 모든 CNN이 고유의 spatial downsampling을 학습할 수 있도록 함. -> 생성자와 판별자 모두 적용

        2. Convolution feature의 가장 top에 있는 full connected layer 제거.

        3. input으로 zero mean, unit variance를 가지도록하는 unit으로 정규화함으로써 학습을 안정화시키는 Batch Normalization을 한다. 하지만 모든 레이어에 batchnorm을 적용하면 sample oscillation과 모델의 불안정성 결과를 얻음 -> 그래서 생성자의 출력 레이어와 판별자의 입력 레이어에는 batchnorm 적용 x

* DCGAN architecture guide line
    
    1) pooling layers -> D는 strided convolution으로 대체하고 G는 fractional strided convolution으로 대체
    2) G, D에 batchnorm 적용
    3) 더 깊은 아키텍쳐르 위해 fully connected 히든 레이어들 제거
    4) G의 모든 레이어에 ReLU를 사용하고 출력 레이어에만 Tanh 사용
    5) D의 모든 레이어에 LeakyReLU 사용

[사진1]

학습 이미지에선 전처리가 적용되지 않았고 tanh activation 함수의 [-1, 1]의 범위로 스케일링 됨. 모든 모델들은 mini batch stochastic gradient(SGD), 미니 배치 사이즈 128로 학습. 모든 가중치들은 zero-centered 정규 분포로 표준 편차 0.02로 초기화 되었다. LeakyReLU에서는, 모든 모델에 leak의 slope는 0.2로 설정되었다. 이전의 GAN 연구들은 학습을 가속하기 위해서 모멘텀을 사용했던 반면, 본 연구에선 조정된 하이퍼파라미터를 사용해서 Adam 옵티마이저를 사용했다. 그 결과 learning rate 0.001은 너무 크다는 것을 발견했고, 대신에 0.0002를 제안. 추가적으로, 모멘텀 beta1을 0.9로 두었을 때 training oscillation과 불안정성이 발견됨. 대신 0.5로 두었을 때 학습이 안정화 됨.

# 6. Inerstigating and Visualizing the Internals of the Networks

# 7. Conclustion and Future work
- GAN 훈련을 위한 안정적인 구조를 제시
- 적대적 네트워크가 생성모델링과 지도학습을 위한 이미지의 좋은 representation을 학습한다는 것을 입증
- 하지만 아직 모델의 불안정성 문제가 남아있음