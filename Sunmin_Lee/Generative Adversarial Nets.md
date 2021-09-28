# Generative Adversarial Nets

# GAN

- 뉴럴 네트워크를 이용한 생성 모델
- Generative : CNN을 이용한 이미지 분류기와 같이 이미지의 종류를 구분하는 것이 아닌 이미지를 만들어 내는 방법을 배우는 모델
- Adversarial : Generative에 적대적인 상대
- Network : 뉴럴 네트워크를 사용해서 모델이 구성되어 있음

## 0. Abstract

- Generative : training 데이터의 분포를 묘사하는 생성모델
- Discriminative : 실제 training 데이터에서 나온 데이터

두 모델이 경쟁하는 과정을 통해 Discriminative Model은 Generative Model의 성능을 최소화하고 Generative Model은 Discriminative Model이 실수할 확률을 최대치로 올리게 만듬

## 1. Indroduction

Deep generative model에서 많아지는 확률 연산들을 근사화하는 것이 매우 어려우므로 이것을 Adversarial nets 즉, 경쟁하는 컨셉을 통해 GAN을 만들게 되었음

## 2. Related Work

### Deep belief networks (DBNS)

- 단일 undirected layer와 몇몇 directed layer를 포함하고 있는 하이브리드 모델
- 장점 : 레이어간 빠르게 근사 가능하다
- 단점 : undirected model 과 directed model이 모두 존재하기때문에 관련하여 계산하기 어렵다

### Auto-encoding variational Bayes

- Back-propagating을 이용한 generative machine 학습 방법

### Stochastic back-propagation

- Back-propagating을 이용한 generative machine 학습 방법

## 3. Adversarial nets

- 두 모델이 multilayer perceptrons일 때 가장 간단하다

![Untitled](Generative%20Adversarial%20Nets%2025d4ce90b8024742bb507176cec3ee48/Untitled.png)

## 4. Theoretical Results

![Untitled](Generative%20Adversarial%20Nets%2025d4ce90b8024742bb507176cec3ee48/Untitled%201.png)

- Generative adversarial nets은 generative distribution(green)을 data generative distribution(black)으로부터 샘플을 구별하기 위해서 discriminative distribution(blue)와 동시에 업데이트하는 것으로 학습된다.
- Mapping 되는 방법
    1. 수렴에 가까운 적대적인 쌍을 고려한다.
    2. Discriminative Model은 수렴하는 데이터로부터 표본을 구별하도록 훈련된다.
    3. Generative Model에 대한 업데이트 이후, Discriminative Model의 gradient는 G(z)가 데이터로 분류된 가능성이 더 높은 영역으로 흐르도록 유도한다.
    4. Discriminative Model이 점점 분류하기 어려워한다
    5. Discriminator가 2개의 분포를 구분할 수 없게 된다.

### Global Optimality of Pg = Pdata

- 이 증명을 통해 minimax problem을 잘 풀기만 하면 generator가 만드는 확률 분포와 pg가 data 분포를 만드는 것과 일치하도록 할 수 있다

Generator가 뽑은 sample들을 Discriminator가 실제와 구분할 수 없다

### Convergence of Algorithm

- 이 증명은 Generator와 Discriminator를 번갈아 가면서 문제를 풀 때 Pg=Pdata를 얻을 수 있는지 확인하는 것이다.

## Experiments

![Untitled](Generative%20Adversarial%20Nets%2025d4ce90b8024742bb507176cec3ee48/Untitled%202.png)

- 시각화한 모델의 샘플들 : 가장 오른쪽 컬럼은 Model이 training set을 기억하지 않는다는 것을 설명하기 위해 가장 근접한 이웃 표본의 training sample을 보여준다.

## Advantages and Disadvantages

### Advantages

- Marko chain이 필요없고 Gradient를 얻기 위해서 back-propagation만 사용된다
- 학습 중 추론이 필요없고 다양한 기능을 모델에 통합할 수 있도록 만든다

### Disadvantages

- Pg(x)에 대한 명시적인 표현이 없다
- Discriminator가 훈련할 때 Generator와 동기화가 같이 잘 되어야한다.

## Conclusions and Future Work

- 조건적인 생성 모델은 Generator와 Discriminator에 c를 input으로 더함에 따라 얻어질 수 있다.
- 학습된 approximate inference는 보조 네트워크를 훈련시킴에 따라서 수행될 수 있다.
- 학습된 대략적인 추론은 z에 주어진 x를 예측하기 위한 보조 네트워크를 훈련시킴으로써 수행될 수 있다.
- Discriminator나 Inference network의 특징은 제한된 레이블이 있는 데이터를 사용할 수 있을때 classifier의 성능을 향상시킬 수 있다.
- Generative Model과 Discriminative Model을 조정하는 더 나은 방법을 세우거나 훈련중에 sample z에 대한 더 나은 분포를 결정함으로써 학습을 크게 가속화할 수 있다.