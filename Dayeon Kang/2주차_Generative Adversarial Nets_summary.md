# Generative Adversarial Nets review

## 0. Abstract
* generatvie model G(생성기) : captures the data distribution
* discriminative D (판별기) : estimates the prob that a sample came from the training data rather than G
* adversarial process : G,D를 동시에 학습
-> G is to maximize the prob of D making a mistake (minimax two-player game)

if G and D are defined by multilayer perceptrons, backpropagation으로 한번에 훈련될 수 있음

----------------------
## 1. Introduction
adversarial nets framework에서 G는 D를 속이도록 setting되고 D는 sample이 G가 모델링한 분포로부터 만들어진 것인지 real data distn로부터 나온 것인지를 결정하는 법을 배운다. 

➡ 이러한 경쟁구도는 두 모델이 각자의 목적을 달성시키기 위해 스스로 개선하는 방향으로 학습된다.

본 연구에서는 MLP로 구성된 G에 random noise를 첨가함으로써 데이터를 생성하고 G도 MLP로 구성된다.

--------------------
## 2.Related work
이전까지의 심층생성모델 연구는 pdf의 parametirc specification을 발견한느 것에 집증되어왔다. (대표적인 예로 Deep Boltzmann machine)
그런데 이런 모델은 일반적으로 gradient에 대한 수치적인 근사가 필요하다.

이러한 어려움으로 적절한 분포로부터 샘플을 생성하는게 가능한 generative machines가 발전하도록하는 모티브가 되었다. 대표적으로 Generative Stochastic Networks가 있다.

-------------------
## 3. Adversarial nets
G의 분포 $p_g$를 x에 대해 학습시키기 위해 input noise 변수에 대한 prior $p_z(z)$를 정의한 뒤 noise변수의 데이터 공간에의 mapping을 $G(Z;seta_g)$로 나타낸다. $D(x)는 입력된 샘플이 $p_g$가 아닌 실제 데이터 분포로부터 얻어졌을 확률을 계산한다.

[사진1]
위의 수식은 D가 true data와 생성데이터에 대해 적절한 label을 할당하도록 하는 prob를 maximize하게끔하고 G가 $log(1-D(G(z)))$를 최소화하도록 훈련시킨다.

* Training process
[2]

        - D가 모델링 하는 conditional prob : 파란색 점선
        - G가 모델링하는 p_g :녹색 실선 
        - 실제 데이터 생성분포 p_x : 검은 점선
        - 하단의 수평선 : z가 균일하게 샘플링되는 도메인
        - 상단의 수평선 : x의 도메인
        - 위로 향하는 화살표 : x=G(z)의 매핑을 통과한 샘플들이 어떤식으로 non-uniform한 p_g를 나타내도록 하는지 보여줌.

=> a,b,c,d 순서대로 보면 G가 모델링하는 녹색 curve는 점점 실제 data distn에 가까워지고 D가 모델링하는 파란색 curve는 점점 1/2에 수렴하는 것을 확인할 수 있다.


## 4. Advantages and Disadvantages
### 1) 장점 
* Markov chain 불필요
* training 단계에서 inference 필요x
* 모델에 다양한 함수들이 통합될 수 있음
* generator network가 데이터로부터 직접적으로 update되지 않고 D로부터 들어오는 gradient만을 이용해 학습 가능


### 2) 단점
* $p_g(x)$에 대한 명시적인 표현이 없음
* training을 진행하는 동안 D와 G가 반드시 균형을 잘 맞춰 학습되어야 함
* 최적해로의 수렴에 있어 이론적 보장의 부족


## 7. Conclusions and future work
1) 클래스 레이블 c를 생성기와 판별기에 모두 추가하는 것으로 조건부 생성모델 p(x∣c)을 얻을 수 있다.
2) x가 주어졌을 때 z를 예측하는 보조 네트워크를 훈련시켜 학습된 근사추론을 진행할 수 있다.
3) 파라미터들을 공유하는 조건부 모델들의 family를 훈련시킴으로써 모든 조건부확률 $p(x_x | x_{nots})$ (s : x의 인덱스들의 부분집합) 를 근사적으로 모델링할 수 있다.
4) 준지도학습 : discriminator에 의해 얻어지는 중간단계 feature들은 레이블이 일부만 있는 데이터셋을 다룰때 분류기의 성능을 향상시킬 수 있다.
5) 효율성 개선 : G와 D의 조정을 위한 더 좋은 방법을 고안하거나 훈련동안 z를 샘플링 하기위해 더 나은 분포를 결정함으로써 학습을 가속화할 수 있음.









