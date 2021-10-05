# 04. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

- supervised learning에 비해 비교적 덜 각광받고, 덜 발전된 CNN의 **unsupervised learning** 해 보기 위해 만든 모델.


- DCGAN의 adversarial pair는 **hierarchy of representations from object part to scenes**, 즉 `계층 구조`를 학습함. 

- 또한, 학습된 feature들을 새로운 tasks에 사용하여 general image representation에 대한 가능성을 입증함.

## I. Introduction

- 당시 large unlabeled datasets에서 feature representation을 학습하는 것이 새롭게 각광받고 있는 연구 분야였음.

- supervised task를 수행하는 데에 `GAN` 이용: GAN을 train한 후, Generator의 부분과 Discriminator NW을 *feature extractors*으로 재사용.
    - feature 추출기로 이용하는 것임.
    - cost function이 없는 process에 대해 매력적임.
    - 그런데 train하는 데에 조금 unstable하고 nonsensical output을 생성하기도 함.

> 이런 맥락에서 `GAN`이 **learn**하는 것이 무엇인지를 알아내고 시각화하는 것, multi-layer GANs의 **intermediate representations**에 대한 연구가 제한적이었음.

그래서 DCGAN 연구자들은 이런 것을 함.

> 1. constraints의 set을 만듦. Conv GAN에게 `제약 조건`을 줘서 안정적으로 훈련할 수 있게 함.
> 
> 2. image classification task에 `trained discriminators`를 이용함. 성능이 다른 것에 비해 competitive.
> 
> 3. GAN에 의해 생성되는 `filters`를 `visualize`함. 그래서 specific filters는 specific objects를 그리도록 학습된다는 것을 보임.
> 
> 4. generator가 `vector arithmetic properties`를 가지는데, 이는 생성된 sample들의 다양한 semantic qualities를 쉽게 조작할 수 있도록 한다는 것을 보임.

## II. Related Work

### II-1. Representation Learning from Unlabeled Data

기존에 unsupervised representation learning을 하는 건 context of image를 찾는 방식이었음. 그런 방식으로는...

- clustering (K-means 같은)
  - leverage the clusters for improved classification scores
  - hierarchical clusturing of image patches → learn powerful image representations
  
- train auto-encoders 
  - separating the what and where components of the code
  - ladder structures
    - encoding: image → compact code
    - decoding: code → reconstruct the image (최대한 정확하게 이미지 재구성하도록.)
  
### II-2. Generating Natural Images

Generative image models도 이제 두 가지 카테고리로 구분됨: parametric, non-parametric

- non-parametric

- parametric
  - blur, diffusion process, suffering from being noisy and incomprehensible, wobbly 등의 문제가 아직도 발생했었음.

### II-3. Visualizing The Internals of CNNs

기존의 CNN은 black-box methods로 사용되어 왔음. networks가 무엇을 form하는지에 대한 이해가 적었던 것. `Zeiler et. al.`은 **deconvolution**과 **filtering the maximal activations**을 이용해서 각각의 convolution filter의 대략적인 의도를 알아냄.

마찬가지로, **gradient descent**를 이용해서 특정 filter 집합을 activate시키는 이상적 이미지를 검사할 수 있음. (inspect the ideal image that activates certain subsets of filters)

## III. Approach and Model Architecture

개인적으로 가장 중요한 것 같은 항목이었음.

기존에 CNN+GAN 시도는 별로 성공적이지 않았음. 그래서 이제 LAPGAN 같은 대안적 접근법이 고안됨. 

연구자들은 또 다른 어려움에 직면했는데, 기존의 CNN+GAN 접근 시도는 거의 supervised learning에만 국한되었다는 것임. 그런데 극복해냄.
이들은 광범위한 모델을 탐색한 후에 의도에 맞는 architectures를 식별했는데, 이건 이제 다양한 데이터셋에 **stable training**을 제공하고 **higher resoultion과 좀 더 깊은 generative model**을 학습할 수 있게 했음.

그래서 이걸 가능하게 한 방법론은 다음과 같음. CNN architecture에 대해 `3가지`를 변경한 것을 채택함.

### 1. Deterministic Spatial Pooling Function → Strided Convolutions 대체

Max Pooling 등의 Deterministic Spatial Pooling Function을 모두 Strided Convolutions으로 대체함. 이렇게 해서 Network가 **own spatial downsampling**을 학습할 수 있게 됨.


### 2. FC Layer 제거

**Convolutional Features 위에 FC Layer을 제거**함. Global Average Pooling은 model stability를 증진시키지만 convergence speed에는 안 좋은 영향을 미침.

Highest Conv Feature을 직접 연결하는 middel ground가 잘 작동했음. 연결은 D & G의 input & output에 함. 모두 잘 작동함.

GAN의 첫 번째 레이어는 unifom noise distribution Z를 input으로 사용함. matrix multiplication이므로 Fully Connected되었다고 말할 수 있지만, 이 결과는 4-dim tensor로 재구성되고 conv stack의 시작이 됨. 

D의 경우, 마지막 conv layer는 flattened되고 single sigmoid output으로 fed됨.

### 3. Batch Normalization

Batch Norm를 이용하면 각 unit에 대해 input들을 모두 **zero mean**과 **unit variance**를 갖도록 정규화시킴. 그래서 학습을 안정화(stabilizes learning)시킴.
이런 안정화는 초기화가 poor할 때 이루어지는 문제를 해결할 수 있고, deeper한 모델에서 gradient flow를 개선시킴.

그리고 이건 deep G가 learning을 시작하는 데에 critical한데, GAN에서 관찰되는 common failure을 시행되지 않도록 함.
이 failure는 G가 all samples를 single point로 collapsing하는 것임. batch norm은 이걸 막아 줌.

그렇지만 모든 layer에 direct하게 Batch Norm을 apply하는 건 sample oscillation과 model instability를 초래하게 됨. ㅠ
이 단점은 이제 **G output layer과 D input layer에 batchnorm을 적용하지 않게** 해서 해결함.

### 4. Activation Function

ReLu Activation은 G에 이용했고, 단 G의 마지막 output은 tahn을 이용함. Bounded activation을 이용하는 것이 model을 좀 더 quick하게 learn하도록 만든다는 것을 발견했기 때문.

그리고 Leaky Rectified Activation은 higher resolution modeling, 즉 고해상도 모델링에 잘 작동한다는 점을 발견했기 때문에 D에 넣음.

이런 activation의 사용은 maxout activation을 이용한 Goodfellow의 GAN과는 대비되는 면임.

---

다음은 논문에서 밝힌 stable DCGAN을 위한 가이드라인임. 정리가 잘 되어 있었음.

> Architecture Guidelines for Stable DCGANs

1. Pooling layers를 **strided convolutions**(discriminator)과 **fractional-strided convoultions**(generator)로 대체함.

2. G와 D에 모두 **batchnorm**을 이용함.

3. **fully connected hidden layers**를 제거함.

4. **ReLU**를 G에 이용함. 그리고 **output에만 tahn**을 이용했음.

5. **LeakyReLU**를 D에 이용함. 이때는 모든 layer에 대해 적용했음.

## IV. Details of Adversarial Training

연구자들은 이 DCGAN을 3개의 dataset에 대해 train해 봄. `LSUN`(Large-scale Scene Understanding), `Imagenet-1k`, `Faces`(newly assembled됐다고 함)이 바로 그것임.

그리고 이 train 과정의 설정은 다음과 같음.

1. scailing을 제외한 pre-processing은 진행되지 않았음.
  - tahn activation function과 같이, [-1, 1]의 범위로 scailing이 진행됨.

2. mini-batch stochastic gradient descent(SGD)로 train됨.
  - 이때 mini-batch size는 128이었음.

3. 모든 weights들은 zero-centered Normal distribution with standard deviation 0.02으로 초기화됨.

4. LeakyReLU에 대해서는, leak의 slope는 0.2로 set됨.

5. hyperparameter tunning에는 Adam optimizer를 이용함.
  - 기존 GAN은 accelerate training에 momentum을 이용한 것과 대비됨.

6. learning rate 0.0002 사용함.
  - 0.0001은 너무 높다고 판단했기 때문임.

7. leaving the momentum term이 0.9이면 training oscillation과 instability가 발견됨. 따라서 **0.5**로 줄여서 training을 안정화시키도록 도움.

### IV-1. LSUN

![image](https://user-images.githubusercontent.com/69420512/135993937-b4951b52-6b2a-4d73-a5ca-ed62fe84e737.png)

이 사진은 LSUN scene modeling에 쓰인 GCGAN Generator 구조임. 
100 dim의 unifom distribution Z는 many feature maps을 가진 small spatial extent convolutional representation으로 투영됨.
그리고 4개의 fractionally-strided convolutions는 이 high level representation을 64x64 pixel image로 바꿈.
이때 연구자들은 또 **fc layer**이나 **pooling layer**가 **사용되지 않았다**는 것을 강조함.

아무튼 이제 LSUN에 적용해 본 걸로 돌아가서~ 생성된 이미지의 퀄리티는 증가했으나 overfitting의 우려와 training sample의 memorization 우려가 높아졌다.

300만 개의 train 사례 결과 **how fast** model learn과 **generalization performance** 둘 사이에는 direct한 연관이 있다고 한다.

fig 2, 3을 보면 이제 우리가 우려했던 overfitting이나 memorization에 의해 이미지가 생성된 것이 아님을 알 수 있음. 그리고 이미지 증강 없었다고 함.

- fig2: small learning rate와 minibatch SGD로 train하기 때문에 memorization 가능성 낮음
- fig3: 5 epoch로 train해서 generate한 이미지. 여러 이미지에서 repeat되는 noise textures가 visual-under fitting 초래함.

#### IV-1-1. Deduplication

이제 앞서 우려됐던 memorization의 확률을 낮추기 위해 **single image de-duplication process**를 수행함. 이미지 중복을 제거하는 프로세스임.

- 3072-128-3072 *de-noising dropout regularized RELU autoencoder*
  - on 32x32 downsampled center-crops of training examples

- resulting code layer activations: binarized via thresholding the ReLU activation
  - has been shown to be an effective information preserving technique
  - provided a convenient form of **semantic-hashing**
    - allowing for linear time de-duplication
  
- hash collisions: high precision with an estimated false positive rate
- technique detected and removed duplicates, suggesting a high recall.

### IV-2. Faces

Random web page에서 많은 face image를 수집해서 수행해 봄. 3K 인간들에 대한 10K 데이터를 수집함. ㅎ 이때 이제 face boxes를 trainig에 수행했다고 함.

### IV-3. Imagenet-1K

Imagenet-1k는 unsupervised learning을 위한 자연 이미지 셋이다. 32x32 min-resized center crops를 사용했다.
여기서도 마찬가지로 데이터 증강 안 했다.

## V. Empirical Validation of DCGANs Capabilities

### V-1. Classifying CIFAR-10 Using GANs As A Feature Extractor

unsupervised representation learning algorithms의 quality를 evaluate하는 방법 중에 하나는 feature extractor로 추가해서 supervised datasets를 이용하는 방법이다.
그리고 이런 feature 위에 fitted된 linear model의 성능을 평가한다.

![image](https://user-images.githubusercontent.com/69420512/136008655-3026bfa6-31ee-4132-a603-370cda1149e6.png)

Table 1. 연구자들이 만든 pre-trained model을 이용한 CIFAR-10 classification. 
DCGAN은 이걸 pre-trained하지는 않았고, Imagenet-1k를 pre-trained함. 그리고 그 pre-trained한 것의 feature들이 CIFAR-10 image를 classificate하는 데에 이용됨. 그때의 성능임.

여기서 연구자들이 강조하는 것은 1. `max # of features unites`가 다른 것에 비해서 많이 적다는 것과 2. `pre-trained`가 CIFAR-10(즉, 같은 데이터셋)에 이루어지지 않았다는 것이다. 
두 번째 강조에서 알 수 있는 것은 **domain robustness of leaning features**를 입증한다는 것이다.

그리고 finetuning을 이용하면 성능 improve가 이루어질 수 있다고 본다는데 그건 후대에 (ㅎㅎ;) 남겨 놓는다고 한다.

### V-2. Classifying SVHN Digits Using GANs as a Feature Extractor

![image](https://user-images.githubusercontent.com/69420512/136009313-48ad390e-ff4f-43ee-a1aa-fbf973f61d3b.png)

Table 2. 1000개의 라벨과 함께한 SVHN classification.. ㅎ SVHN은 StreetView House Numbers dataset이다. 

여기서는 동일한 데이터, 동일한 아키텍처로 이루어진 CNN을 교육/최적화해서 DCGAN이랑 함께 비교했다. 보면 이제 CNN이 모델 성능을 높이는 데에 key contributing factor가 아니라는 것을 입증한다.

## VI. Investing and Visualizing The Internals of the Networks

이 연구자들은 G와 D를 train할 때 아무런 nearest neighbor search도 하지 않는다.

### VI-1. Walking in the Latent Space

- understand the landscape of the latent space
  - walking on the *manifold that is learnt*
      - *memorization*의 sign에 대해서 우리에게 이야기해 준다. 
      - 어떤 space가 *hierarchically collapsed* 하는지, 그 way 또한 이야기해 준다.
  - 그래서 만약 walking in this latent space가 semantic changes를 초래한다면, 우리는 이게 모델이 relevant and interesting representations를 학습해서 그렇구나~ 알 수 있다고 한다.

- Fig 4. Z의 9개 random points 사이의 **Interpolation**은 모든 이미지가 smooth trasitions를 가지고 있음을 보여 준다.
  - 6번째 row에서는 `창문이 없는` 방이 `창문이 있는` 방으로 천천히 변한다.
  - 10번째 row에서는 `TV`가 `창문`으로 변한다.

### VI-2. Visualizing The Discriminator Features

CNN supervised learning의 선례 연구는 very powerful features를 보인다. 
추가로, supervised CNN 중 scene classification에 의해 훈련된 친구는 object detectors를 learn한다.

이와 마찬가지로 연구자들이 고안한, large image dataset에 의해 trained된 DCGAN 또한 interesting한 hierarchy of features를 learn할 수 있다고 한다.
**guided backpropagation**을 이용하면, D에 의해 학습된 특징들이 약간 typical한 부분에서 activate한다는 것을 알 수 있다.

- Fig 5. previous responses과 비교했을 때, discrimination과 random structure가 거의 없거나 진짜 없음.
  - beds의 features에 굉장히 significant minority에 반응함. 

### VI-3. Manipulating the Generator Representation

#### VI-3-1. Forgetting To Draw Certain Objects

D에 의해 learn되는 representations에 추가해서, G가 그럼 어떤 representation을 learn하냐? 라는 질문이 있을 것이다.

samples의 quality는 이제 G가 **major scene components**의  specific object representation을 학습한다는 것을 암시한다.

그래서 연구자들은 G에서 침대를 제거하는 실험을 했다. 이때 이제 second highest conv layer features에 대해, **logistic regression을 이용해서 weight가 0보다 큰 all feature map을 dropped** 시켰다.
그리고 feature map removal 없이 random new sample들을 제작했다.

window dropout with and without 이미지는 fig 6에서 나타나 있고, nw는 windows를 거의 잊어버리고 other objects로 그것들을 대체한다고 한다.

![image](https://user-images.githubusercontent.com/69420512/136013386-eb776d26-5163-4919-8f9f-7269d5988a2b.png)

- Fig 6. G가 **object** represention과 **scene** representation을 잘 구별했다는 것을 알 수 있음.
  - Bottom row: the same samples generated with dropping out "window" filters.
  - 어떤 window들은 removed 되었는데, 이제 doors나 mirrors 같은 similar한 visual appearance로 대체됨.
  

#### VI-3-2. Vector Arithemetic on Face Samples

Mikolov의 연구에서 representations of words가 간단한 arithmetic operations에서 **rich linear structure**를 나타냈다는 것을 알 수 있었다.
그러니까 `vector(King) - vector(Man) + vector(Woman)`이라는 vector에 가장 가까운 결과는 `vector(Queen)`이라는 것을 증명한 것이다.

**DCGAN의 Generator의 Z에서도 동일한 구조가 나타나는지 조사**했고, 유사한 연산을 시행했다. only single samples에 대해 적용하는 실험은 불안정하기는 했는데, 그래도 averaging the Z vector for three samples는 *consistent*하고 *stable* generations을 보여 줬다.

우리의 모델에 의해 learn된 Z representation을 이용해서 흥미로운 응용들을 시행해 볼 수 있다. 특히 이 모델은 convincingly model object attributes들을 learn 할 수 있다는 것을 입증했다.

![image](https://user-images.githubusercontent.com/69420512/136014979-49782a06-a908-4acd-94ca-7d388bf24ac7.png)

- Fig 7. Object Manipulation

![image](https://user-images.githubusercontent.com/69420512/136010909-97e5291e-6126-44c5-9722-df5311055880.png)

- Fig 8. Face pose가 Z space에서 modeled linearly 됐다.
  - left한 쪽을 보고 있는 sample과 right sample들의 average에 의해 `turn` vector가 만들어진다. 
  - interpolations along this axis to random samples에 따라 이 pose를 조금 더 잘 transform할 수 있다고 한다.
  
vector archmetic은 complex image distribution의 conditional generative modeling에 **필요한 데이터의 양**을 극적으로 줄일 수 있다.

## VII. Conclusion and Future Work

GAN을 train하는 데에 조금 더 **stable**한 architecture을 고안했다. 또한 adversarial nw이 **good representations**을 learn하는 것을 알 수 있다.
supervised learning, 그리고 generative modeling을 위한 좋은 표현들을 학습하는 것이다.

그런데 이제 instability한 것이 남아 있다고 한다. models의 train이 길어짐에 따라 가끔은 **collapse a subset of filters**해서 **a single oscillating mode**로 바꾼다는 것이다.

그래서 앞으로의 연구는 이런 instability를 타파해야 한다. 또한 이걸 image뿐만이 아니라 audio나 video에 적용시키는 것도 재밌을 거다. ㅎ

learnt latent space의 properties에 대한 investigation 또한 재밌을 거라고 한다. 글쿤....