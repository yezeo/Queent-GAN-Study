# 03. Conditional Generative Adversarial Nets

기존 GAN 구조는 noise로부터 구현된 이미지를 **제어할 수 없었**지만, **CGAN**은 **condition을 부여/제어**하여 원하는 이미지를 만들고자 한다.


이때 이 condition을 부여하려고 기존 GAN 구조에 간단하게 데이터 y를 추가해 줌으로써 구현한 모델. Generator와 Discriminator 둘 모두에 condition을 부여했다.

논문의 검증으로는 MNIST를 사용했는데, label을 condition을 두었고 그에 따라 원하는 label(condition)의 이미지를 잘 생성하는 것을 볼 수 있다.

이런 기법을 이용해서 multi-modal model을 학습시킬 수 있고 image tagging 심화에도 사용할 수 있다. Training label이 아닌 **descriptive tags**를 만들 가능성을 열어 준다.

## Introduction

GAN을 심화시킨 모델이다. GAN은 1) obtain gradients에 backpropagation만이 필요하고, 2) learning 와중에 inference가 필요하지 않으며, 3) wide variety of factor and interactions가 모델에 그냥 적용된다는 이점이 있다. 또한, 4) 실제적인 모델과 art log-likehood estimates를 생성할 수 있다.

그런데 이제 이것이 *unconditional* 모델이라 data의 **mode에 대한 설정이 불가능**하다. 그래서 additional information에 대해 condition을 설정해 주면 data generator process를 우리가 direct 할 수 있다. 그러니까 어떤 이미지를 생성할지 조건을 넣어 줄 수 있는 것이다.

이런 조건 설정은 class label을 이용할 수도 있고, 다른 modality에서 온 data를 이용할 수도 있다.

## Related Work

### 2-1. Multi-modal Learning For Image Labelling

지금까지의 연구는 문제점이 있는데, 일단 두 가지로 간추리면,
1. Challenging to scale such models to **accommodate an extremely large number of predicted output categories.**
   
2. Focused on learning **one-to-one mappings** from input to output.

이 두 가지 문제점이 있다. 그래서 일단 첫 번째 문제를 해결하는 방법부터 보자.


> 1. Challenging to scale such models to **accommodate an extremely large number of predicted output categories.**


Leverage additional information from other modalities해서 발전시킬 수 있다. Image feature-space를 word-representation-space를 단순히 linear하게 연결(mapping)하는 것만으로도 큰 발전이 된다는 사실.


> 2. Focused on learning one-to-one mappings from input to output.

이건 conditional probabilistic generative model을 이용해서 해결할 수 있다. input은 conditioning variable이 되고, one-to-many-mapping이 conditional predictive distriction이 되면 된다.

## Conditional Adversarial Nets

### 3-1. Generative Adversarial Nets

![image](https://user-images.githubusercontent.com/69420512/135067883-96e3dc05-1e64-4c57-858c-d14f962339fe.png)

일반 GAN에 대해 설명한다. G, D가 동시에 train되고... 마지막 log값을 min하기 위해 G의 파라미터를 조정하고 첫 번째 log를 min하기 위해 D의 parameter을 조정한다. 그래서 V(G,D)를 조정하기 위한 two-player min-max game을 한다고... 보면 된다.

### 3-2. Conditional Adversarial Nets

이건 그냥 위 GAN 모델에 condition을 부여하기 위해 **extra information y**를 부여한다. y는 뭐... 아무 auxiliary information이 다 될 수 있다. other modalities나 label 등등. 
그래서 여기서는 단순히 **feeding y into the both the discriminator and generator as addition input layer** 해서 condition을 부여할 수 있다. 여기서 알 수 있듯
G와 D 모두에 y를 넣어 주었다. 수식은 위 GAN에서 x, z였던 게 x|y, z|y 꼴이 된다.

![image](https://user-images.githubusercontent.com/69420512/135067956-ee478a23-821d-4a39-8680-012b5d80fe18.png)

generator에서, 앞선 연구의 input noise는 pz(z)였는데 y가 **combined in joint hidden representation**이 된다.

![image](https://user-images.githubusercontent.com/69420512/135068665-363b992b-eead-4e5e-b352-c648e3bacb8a.png)

그림으로 표현하면 이렇게 됨. 원래는 각각 걍 z랑 x였던 것이 z|y, x|y로 conditional하게 바뀐 것을 알 수 있음.

## Experimental Results

### 4-1. Unimodal: MNIST

MNIST digits를 generate하는 데에 label이라는 condition을 걸어 줬다. 아래 사진에서 Each row가 one label에 conditioned 됐고 each column은 다 different gerated sample을 가진다.

연구 결과, 다른 것으로 한 것보다 훨씬 결과가 잘 나왔다고 한다.

![image](https://user-images.githubusercontent.com/69420512/135069025-03afe31f-0ee2-439c-849d-8665d769b566.png)

### 4-2. Multimodal

여기서는 multimodal에 대해서도 해 본다. **automated tagging of images with multi-label predictions**를 해 봤는데, **이미지 자동 태깅**에 대한 것이다.

이 연구자들은 사람들이 사진을 올려 놓는 사이트에서 소스를 가지고 왔는데, 사람들이 직접 올린 것이다 보니까 그냥 identify the object present an image하는 것보다는 more descriptive하고, 인간이 이미지를 표현하는 방식과 좀 비슷하다. 
또다른 UGM의 특징은 사람들이 같은 이미지, 같은 것을 표현하더라도 different vocabulary를 쓸 것이기 때문에 having an efficient way to normalize these labels하는 것이 중요해진다. 그래서 Conceptual word embedding이 유용하다고 한다.

image features는 CNN으로 사전 교육하고, user-tages, titles, descriptions에서 corpus of text를 가지고 온다. (여기서는 YFCC100M 2를 썼다고 한다.)

![image](https://user-images.githubusercontent.com/69420512/135070300-863c4460-9449-403d-9ad9-065a154045c4.png)
![image](https://user-images.githubusercontent.com/69420512/135070377-a348df00-48b9-4768-b225-6aae11e79a42.png)

Generative tags들은 이런 느낌인데 여기서 마지막 사진에 love... 이런 거 나온 거 너무 신기하다.

## Future Work

지금은 condition에 simple tag를 썼지만 나중에는 multi tag도 쓸 수 있을 것이고 언어 모델을 학습할 수 있을 것이라는 가능성 정도를 열어 두었다.

