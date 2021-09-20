본 글은 Ian J. Goodfellow가 2014년에 발표한 [GAN(적대적 생성 신경망)](https://arxiv.org/abs/1406.2661) 논문을 참고해서 작성했으며, 제 주관적 견해와 정보들이 정리되어 있습니다.

Paper Review에 들어가기 전 GAN의 역사와 현황에 대해 간략히 말하자면,

GAN이 최초로 제안된 2014년 이후로 수많은 파생 GAN들이 잇따라 발표되었으며 그 개수가 너무나도 많아(..ㅠ) 따라갈 수 있을지 의문이 든다.. 

GAN을 필두로 DCGAN, LSGAN 등등 GAN 역사에 한 획을 그은 연구들을 중심으로 더 리뷰할 예정이다!

다양한 GAN들을 트랙킹하고 싶다면 ☞☞☞ [The-Gan-Zoo](https://github.com/hindupuravinash/the-gan-zoo)

---

## 0\. Abstract

논문의 저자는 경쟁하는/적대적인 관계를 가지고 있는 두개의 모델을 동시에 훈련시키는 새로운 프레임워크를 제안한다.

두 모델은 각각 G : Generator, D : Discriminator , 생성자와 판별자이다.

한마디로 하면, 이미지를 만들어내는 사람 vs. 이미지를 구별하는 사람 정도로 생각해 볼 수 있다.

\- 생성자(이하 G)는 랜덤 노이즈 벡터를 입력받아 데이터를 생성한다.

\- 판별자(이하 D)는 특정 데이터가 가짜인지 진짜인지를 판별한다.

이 때 G는 더 진짜같은 이미지 데이터를 생성하도록 훈련받고, D는 진짜와 가짜를 더 잘 구별해낼 수 있게 훈련받는다.

임의의 함수(arbitrary function) G, D의 공간에서, G는 training 될수록 training data를 모사하게 된다.그리고 D가 실제 training 데이터인지 G가 생성해낸 fake 데이터인지 판별하는 확률값이 1/2가 될 때 최적의 해라고 저자는 말한다.

## 1\. Introduction

기존의 딥러닝은 풍부한 parameter들과 계층구조가 잘 나타나게 설계된 모델을 사용해, 다양한 데이터 타입(이미지, 자연어, 음성, etc.)의 분포를 표현하는데 의의가 있었다. 그 중에서도 classification과 같은 discriminative model이 성과 측면에서 두곽을 나타냈다. 그 이유는 바로 backpropagation과 dropout과 같은 알고리즘으로 gradient들을 성공적으로 다룰 수 있었기 때문이다.

하지만 Deep Generative 모델은 좀 다르다. 기존의 생성 모델은 computation이 다루기 복잡했고, 활성화 함수들의 이점을 충분히 활용할 수 없는 구조였기 때문에 각광 받지 못했다. 이 문제점을 해결한게 바로 GAN이다. GAN에서는 forward propagation, backward propagation, dropout을 생성모델에서도 사용 가능하게 했다.

[##_Image|kage@n19XK/btq8olcbkuo/uvaGYXMoeTwK3u2G3XvXC1/img.png|alignCenter|data-origin-width="1280" data-origin-height="463" width="799" height="289" data-ke-mobilestyle="widthOrigin"|||_##]

논문에서는 GAN을 위조지폐범과 경찰의 경쟁, 대립에 비유했다.

-   위조지폐범은 최대한 진짜 같은 화폐를 만들어 경찰을 속이기 위해 반복 학습하고, 경찰은 진짜 화폐와 가짜 화폐를 완벽히 가려내는 것을 목표로 한다. 이런 경쟁 과정을 반복하다보면 어느 순간 위조지폐범의 스킬이 상당해져 진짜같은 가짜를 만들어내고, 경찰은 이를 구분해내지 못하면서 경찰이 위조지폐를 구별할 수 있는 확률이 50%로 수렴하게 된다.

여기서 중요한 점은 서로 다른 목표를 가진 두 모델이 경쟁하면서 위조지폐범이 점점 ‘_진짜같은 가짜_’를 생성해낼 수 있게 된다는 점이다.

## 3\. Adversarial Nets

[##_Image|kage@cke8WA/btq8q1c7UDC/TBfiq1A1tnbxDrcVQeInUK/img.png|alignCenter|data-origin-width="1498" data-origin-height="692" width="639" height="295" data-ke-mobilestyle="widthOrigin"|||_##]

| 기호 | 의미 |
| --- | --- |
| $x$ | Real Data |
| $z$ | noise, 확률분포로부터 추출한 샘플 z, 가짜 이미지를 생성할 '재료' |
| $p\_g$ | $x$에 대한 G의 분포 |
| $G(z)$ | G가 Noise z를 받아서 생성한 Fake Data. (Real data와 사이즈가 같아야함) |
| $\\theta\_g$ | multilayer perceptrions의 parameters |
| $G(z;$$\\theta\_g)$ | data space에 대한 mapping |
| $D(x)$ | $x$가 $p\_g$가 아니라 원본 데이터에서 나왔을 확률 |

\- G는 noise z를 input으로 받아 $G(z)$ 를 output한다. Input z는 보통 zero-mean, identity-covariance multivariate Gaussian N(0,1) 으로부터 샘플링되는데 그냥 간단하게 말해서 Gaussian Distribution으로 노이즈를 생성한다. 위 그림에 예시가 나와 있다. Output $G(z)$ 는 주로 이미지이다.

  
\- D는 real data $x$ 와 synthetic data $G(z)$ 를 input으로 받아서 확률을 output 한다. output은 진짜일 확률로, 기본적으로 0과 1사이의 단일 스칼라 값이다.

[##_Image|kage@bkNz2X/btq8sDoMSKH/lHDKWzmem5tfT8eJPcKikK/img.png|alignCenter|data-origin-width="599" data-origin-height="41" data-filename="blob" data-ke-mobilestyle="widthOrigin"|V(D,G) : value function||_##]

Value Function에 왜 log를 쓰는지에 대해 생각해보았다. 아마도 D(x)의 출력값이 0-1사이의 소수값이어서 log를 씌워서 scailing을 확대시켜주기 위해서인거 같다. 그리고 log가 정규분포를 만드는데도 더 유리할거 같다. 

Value function V를 각각 G와 D의 관점으로 나눠서 생각해보면 서로 다른 목적을 갖고 있음을 알 수 있다.

G의 입장 : G는 두번째 항 $log(1-D(G(z)))$에만 관여한다.

G는 D를 속이는게 목표이기 때문에 $D(G(z))$= 1이길 바란다.

→  $log(1-D(G(z))) = log(1-1) = log(0) = -\\infty $.  즉, V(G,D)가 최소값을 갖는다.

$$Loss\_G = Error(D(G(z)),1)$$

D의 입장 : D는 첫번째 항, 두번째 항 모두 관여한다.

D는 최대한 맞게 판별하는게 목표이기 때문에

$logD(x) = log(1)$ ,   $log(1-D(G(z))) = log(1-0) = log(1)$

즉, 첫째항 둘째항 전부 log1로 V(G,D)가 최대값을 갖는다.

$$Loss\_D = Error(D(x),1) + Error(D(G(z)),0)$$

※ Error함수는 다양하게 선택 할 수 있는데 Cross-Entropy Error가 주로 쓰이고 L2, bce도 쓰인다.

---

V(D,G)에서 문제점이 두가지가 있는데,

1\. D를 학습시킬때 computation의 부담이 굉장히 크고, 유한한 데이터셋 x를 가졌을때 overfitting이 될 수도 있다. 그래서 논문에서는 하나의 epoch를 돌릴 때, D를 k번, G를 1번 training 시킨다. (뒤에 algorithm에서 더 자세히 나온다)

[##_Image|kage@czt87X/btq8me5zIn8/G2GovYtnlBtY2muOvTokhk/img.png|alignCenter|data-origin-width="871" data-origin-height="585" width="452" height="304" data-ke-mobilestyle="widthOrigin"|[Algorithm 1]||_##]

2\. 초기 모델 G는 아직 파라미터들이 학습이 제대로 안된 상태이기 때문에 D의 입장에서 봤을때 G(z)를 판별해내기란 너무 쉽다. 완전히 터무니 없는 값을 D에 input 하기 때문이다. 이렇게 되면 $log(1-D(G(z)))$가 saturated(포화)되고, _학습이 정체된다._

_☞ 초기에 G의 성능이 나쁠 때에는 $log(1-D(G(z))$의 gradient를 계산했을 때 너무 작은 값이 나오므로 학습이 느리기 때문_

해결방법 : $\_\\min log(1-D(G(z)))$ 를 $\_\\max logD(G(z))$로 바꾸면 된다. 목적은 같고 방법만 달라진다.

[##_Image|kage@RDH0j/btq8pitaDTQ/1pdtcEnfRsaxhlP8x7KVbK/img.png|alignCenter|data-origin-width="1296" data-origin-height="486" width="602" height="226" data-ke-mobilestyle="widthOrigin"|출처 :&nbsp;https://hyeongminlee.github.io/post/gan001_gan/||_##]

그래프를 보면 이해가 조금 쉬울거 같은데, 같은 1에서의 기울기라도 log(x) 와 log(1-x)에서의 gradient가 확연히 다르다. 오른쪽 그래프를 사용하면 gradient vanishing problem을 해결 할 수 있다.

## 4\. Theoritical Results

[##_Image|kage@b38ngi/btq8rXHNMFg/31aAdJC3aAAZqJx9pDk5l0/img.png|widthContent|data-origin-width="500" data-origin-height="205" data-ke-mobilestyle="widthOrigin"|알고리즘 진행과정에 따른 G와 D의 분포 변화||_##]

위 그림은 알고리즘이 진행함에 따라 G와 D의 분포가 어떻게 변하는지 보여준다. 이해를 돕기 위해 논문에서는 간단하고 축약적으로 분포를 나타냈지만, 실제 데이터는 차원이 매우 크기 때문에 그림 같이 이차원으로 나타낼 수는 없다.

(a) : 알고리즘 실행 전, _D의 분포가 불규칙_하다. 그리고 _Real data 분포와 G의 분포가 완전 다르다._

(b) : k번 D를 학습시킨 결과. D가 아까보다 훨씬 더 안정적으로 fake 와 true data를 구별하는것을 볼 수 있다.

(c) : 어느정도 학습이 된 D는 고정해놓고, G를 한번 학습. 분포가 얼추 비슷해져 간다.

(d) : b와 c가 몇차례 반복되면서 G와 분포와 실측데이터 분포가 일치(Pdata = Pg)하게 되고, 이때 D 분포는 uniform함을 보이게 되며, 1/2에 수렴한다.

---

이제부터는 굉장히 중요한 부분인 **수식유도**가 나온다. 내용이 길어서 자세한 증명은 아래 게시물을 확인하길 바란다.

[2021.06.29 - \[Deep Learning/GAN\] - \[GAN\] Generative Adversarial Nets - 수식 증명](https://memesoo99.tistory.com/27)

GAN의 목적을 다시 떠올려 보면, $P\_g$ 와 $Pdata$를 유사하게 만드는 것이다.

다시 말해, Pdata와 Pg의 거리가 최소화될 수 있도록 만들어주는 것이다.

이를 뒷받침할 증명 두개가 있다.

### 4.1 Global Optimality of $P\_g = Pdata$

: GAN의 Minimax problem이,  $(P\_g = P\_data)$ global optimum을 갖는가?

### 4.2 Convergence of Algorithm 1

: 논문에서 제안하는 Algorithm이 실제로 global optimality $(P\_g = P\_data)$을 찾을 수 있는가?

## 5\. Experiments

MNIST로 GAN을 돌린 결과. Noise부터 시작해서 점점 그럴싸해져가는 과정이 신기하다. 

[##_Image|kage@p35z7/btq8nsXlpdW/eqiUo4zNw9gzxzFmWinhJk/img.gif|alignCenter|data-origin-width="646" data-origin-height="642" data-filename="conv-mnist.gif" width="482" height="479" data-ke-mobilestyle="widthOrigin"|출처 :&nbsp;https://github.com/TengdaHan/GAN-TensorFlow||_##]

\[참고\]

 [\[GAN\]Generative Adversarial Network | Hyeongmin Lee's Website

이번 포스트에서는 GAN의 기본 개념과 원리에 대해 알아보도록 하겠습니다. GAN(Generative Adversarial Network)은 Generator와 Discriminator의 경쟁적인 학습을 통해 Data의 Distribution을 추정하는 알고리즘입니

hyeongminlee.github.io](https://hyeongminlee.github.io/post/gan001_gan/)

 [GAN tutorial 2016 정리(1)

GAN tutorial 2016 내용 정리. GAN tutorial 2017 ( 이 나온 마당에 이걸 정리해본다(..) \_소개. Generative model들중 어떤 아이들은 density estimation을 통해 generate한다. (like variational inference autoencoder) 어떤 data-gene

kakalabblog.wordpress.com](https://kakalabblog.wordpress.com/2017/07/27/gan-tutorial-2016/)