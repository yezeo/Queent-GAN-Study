# 02. Generative Adversarial Nets

Adversarial Modeling framework의 실현 가능성을 입증해서 연구 방향을 제시한 논문임.

## Abstract

> GAN은 생성 모델 G와 판별 모델 D 두 개로 이루어진다. G는 D가 실수를 만들도록(가짜 이미지를 진짜 이미지로 판별하도록) train된다. 
이떄 GAN 모델은 multilayer perceptron으로 이루어지며, 전 과정은 backpropagation을 통해 train될 수 있다. (선행 연구와 다르게 Markov Chain이나 unrolled approximate.. 필요 없음)

- A generative model **G**
    - captures the *data distribution*  
    - training procedure for G: to **maximize** the probability of *D making a mistake*
    

- A discriminative model **D**
    - *estimates the probability* that a sample came from the training data rather than G


- Correspond to a **minmax two-player game**


- G and D
    - defined by *multilayer perceptron*
    - entire system can be trained with *backpropagation*
        - no need for any Markov chains
        - no need for unrolled approximate inference networks
    
    

## Introduction

 > 그 유명한 경찰-도둑 게임 비유. 도둑이 위조 화폐를 만드는데, 게임이 진행될수록 이 화폐가 가짜인지 진짜인지를 구별할 수 없을 정도로 성능이 improve된다.

- Competition in this game: drives both teams to improve their methods until the counterfeits are indistinguishable from the genuine articles.
    - G: counterfeiters
        - trying to produce fake currency and use it without detection
      
    - D: the police
        - trying to detect the counterfeit currency

> 이 프레임워크는 다양한 모델과 optimization을 사용해서 다양한 알고리즘이 될 수 있는데 이 논문에서는 둘 다 multilayer perceptron일 경우를 다룬다.
특히 G가 다층 퍼셉트론을 이용해 랜덤 노이즈를 전달받는 경우에 대해 다룬다.

> 그리고 그런 구조를 adversarial nets라고 하는데 굉장히 성공적인 backpropagation만을 이용해서 train할 수 있고,
generative model 샘플을 만드는 데에도 forward propagtation만 필요하다.
(아마 예전 연구에서 많이 쓰인 것 같은) Markov chains.. 같은 건 필요없다고 한다.

- Can yield specific training algorithms for many kinds of model and optimization algorithm.
    - **adversarial nets**
        - we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron
        - and the D is also a multilayer perceptron
    - In this case, can train both models using *only the highly successful backpropagation and dropout algorithms*
    - and sample from the generative model using *only* forward propagation.
    - No approximate inference of Markov chains are necessary.

## Related Work

> 과거 모델에는 numerous approximations이 필요하다는 단점이 있었다. 그래서 이런 단점이 generative machine을 만드는 것의 motivation이 되었고
> 그래서 오로지 **backpropagation으로만으로도 train**할 수 있는 `Generative stochastic networks`가 만들어지게 된다.
> 이때 distribution을 명시적으로 나타내지는 않지만, 원하는 분포로 샘플을 생성해낼 수는 있다고 한다.
> 여기에서는 **Markov chains**라는 게 쓰이는데, 이번 연구는 그것을 **제거**하여 generative machine 아이디어를 조금 더 강화했다.

- 과거: model that provided a parametric specification of a probability distribution function.
    - can be trained by maximizing the log likelihood (ex. deep Boltzmann machine)
    - difficulty: generally have intractable likelihood functions → *require numerous approximations* to the likelihood gradient
    - → motivated **the development of *generative machines***
    - models that do not explicitly represent the likelihood, yet are *able to generate samples from the desired distribution.*
    

- Generative stochastic networks: a generative machine that can be *trained* with *exact backpropagation*
    - rather than the numerous approximations required for Boltzmann machines
    

- This work extends the idea of a generative machine by **eliminating the Markov chains** used in GSN
    
> GAN은 아래 식과 같은 observation을 이용하여 generative process를 backpropagate한다.

![image](https://user-images.githubusercontent.com/69420512/133883788-ef3d96eb-0a06-47db-b1b4-1cd31f5593b3.png)


- `general stochastic backpropagation rule`
    - Allowing one to backpropagate through Gaussian distributions with finite variance,
    - and to backpropagate to the covariance parameters as well as the mean
    - could allow one to *lean the conditional variance of the generator* (hyperparameter)
    
> `VAE`라는 모델은 GAN과 마찬가지로 2개의 네트워크를 가진다. (Generator Network) 
> 그런데 GAN과 달리 두 번째 네트워크는 대략적 추론을 수행한다. 그래서 Discrete Latent Variables를 가질 수 없다.
> GAN은 discrete data를 모델링할 수 없다.

- `VAE`(variational autoencoders)라는 모델은 train 과정에서 이 stochastic backpropagation을 사용했다.
    - *like GAN*: variational autoencoders pair a differentiable generator network with a second nn
    - *unlike GAN*: the second network in a VAE is a recognition model that performs approximate inference
        - GANs require differentiation through the visible units → cannot model discrete data
        - VAEs require differentiation through the hidden units → cannot have discrete latent variables
    
> 이전 연구들은 discriminataive criterion을 이용해서 generative model을 train했다. 그런데 이건
> 깊은 모델을 training시키는 데에 적합하지 않다. 왜냐면 여기에는 approximate하기 힘든 확률의 비율이 포함되기 때문이다.
> 그래서 `NAE`(Noise-contrasive estimation)를 이용했다.

> `NAE`는 generative model를 training할 때 weights를 학습하는데, 이 weights는 fixed noise distribution에서
> 추출되었으며 data를 discriminate하기에 유용하다. 그래서 이런 노이즈 분포를 이용하면 성능이 향상된다.
> 그런데 여기서의 한계는 discriminator의 정의이다. 한쪽만 가능하므로 **양쪽의 밀도를 통해 모두 평가하고 역전파**해야 한다.

- NAE
    - “discriminator” is defined by the ratio of the probability densities of the noise distribution and the model distribution
    - requires the ability to evaluate and backpropagate through both densities. Some previous work has used the general concept of having two neural networks compete.
    
> 그래서 양쪽 밀도를 통해 모두 훈련하는 방식을 찾아야 하는데, Predictability Minimization이 그것이다.
> 그런데 이제 이 연구와 GAN이 다른 점은 크게 3가지가 있다. 1) GAN에서는 **Competition btw the NW**이 유일한 train 기준이다. 2) competition의 성격 자체가 다르다. 3) learning process의 specification이 다르다.
> P.M은 최적화에 중점을 두는 반면 GAN은 minmax 게임이 기본이다.

- Predictability Minimization
    - each hidden unit in a neural network is trained to be different from the output of a second network, which predicts the value of that hidden unit given the value of all of the other hidden units.
    - **Differ in GAN**
        -  1) the competition between the networks is the sole training criterion
            - sufficient on its own to train the network.
            - P.M은 정규화를 하기 위해 권장할 뿐 필수는 아님
        - 2) the nature of the competition 자체가 다름
            - P.M: two networks’ outputs are compared, with one network trying to make the outputs similar and the other trying to make the 2 outputs different. The output in question is a single scalar.
            - GAN: one network produces a rich, high dimensional vector that is used as the input to another network, and attempts to choose an input that the other network does not know how to process.
        - 3) The specification of the learning process is different.
            - P.M: described as an optimization problem with an objective function to be minimized, and *learning approaches the minimum of the objective function.*
            - GAN: based on a minimax game (than optimization problem)
                - have a value function that one agent seeks to maximize and the other seeks to minimize.
                - terminates at a saddle point: a minimum with respect to one player’s strategy and a maximum with respect to the other player’s strategy.
    
> 그리고 GAN은 Adversarial Example랑 다른 개념이다.

## Adversarial Nets

- The adversarial modeling framework: most straightforward to apply when the models are both *multilayer* perceptrons.
    - To learn the generator's distribution `p_g` over data `x`
        - define a prior on input noise variables p_z(z)
        - represent a mapping to data space as G(z;Θ_g)
            - where G is a differentiable function represented by a multilayer perceptron with parameters Θ_g
        - define second multilayer perceptron D(x;Θ_d) that outputs a single scalar
    
> 내가 생각하기에는 여기가 핵심인 것 같다. D(x)는 x가 G와 다른 분포의 데이터일 확률, 즉 가짜 이미지일 확률을 나타낸다.
> 따라서 우리는 D를, train data와 **생성된 데이터에 "진짜 이미지"** 라벨을 붙일 확률이 **최대**가 되는 방향으로 train한다. 
> 그리고 G를, log(1-D(G(z)))가 **최소**가 되는 방향으로 train한다.

- D(x): represents the *probability that x came from the data rather than p_g*(p_g: generator's distribution)
    - We train D → *maximize* the probability of assigning the correct label to both training examples and sample from G.
    - We simultaneously train G → *minimize* log(1-D(G(z)))
    
> 이런 건 이제 앞서 말했듯 도둑-경찰 게임으로 표현할 수 있고, two-player minmax game이라고 이름을 붙이고 value function을 아래와 같이 정의할 수 있다.

- D and G play the following *two-player minmax game* with value function V(G,D):
  

  ![image](https://user-images.githubusercontent.com/69420512/133887273-3b8f0a62-f0c9-42a2-9c76-cd0c11345fe0.png)

> Training은 어떤 과정을 거쳐 진행될까? 하단 사진은 간단하게 묘사해 주고 있다.
> 
> `The training criterion`은 G와 D에 충분한 데이터가 주어졌을 때 데이터 분포를 복구할 수 있게 한다.
> 그래서 이 복구를 위해서는 반복적이고 수치적인 접근법을 이용해야 한다. (위 식처럼~)
> training 과정에서만 D를 optimize하는 것은 overfitting이나 제대로 되지 않은 결과를 초래할 수 있다.
> 그래서, D optimization을 k번 할 때마다 **중간에 G optimization도 한 번씩 해 줘야 한다.** (번갈아 해 줘야 된다는 소리다.)
> 이렇게 해 주면 D는 이제 optimal solution이랑 근접하게 진행되고, G는 충분히 느리게 진행된다. (k번-1번 대응이니까)

- The training criterion allows one to recover the data generating distribution as G and D are given enough capacity i.e., in the non-parametric limit.
    - In practice, we must implement the game using an iterative, numerical approach.
    - Optimizing D to completion in the inner loop of training: computationally prohibitive, and on finite datasets would result in overfitting
    - Instead, we alternate between *k* steps of optimizing D and one step of optimizing G.
    - The results in D being maintained near its optimal solution
    - G changes slowly enough

![image](https://user-images.githubusercontent.com/69420512/133888290-09d13f64-50f5-44ba-ba84-e47fb72188c4.png)
(less formal, more pedagogical explanation of the approach)

> GAN은 검은색-녹색이 구별되도록 discriminative distribution을 동시에 업데이트하여 train함
> 화살표는 x=G(z) 변환 표본에 불균일 분포 pg를 부과하는 방법 보여 줌. 

-  Generative adversarial nets are trained by simultaneously updating the discriminative distribution (D, blue, dashed line) so that it discriminates between samples from the data generating distribution (black, dotted line) px from those of the generative distribution pg (G) (green, solid line).

> 여기서 학습 초기에는 G가 형편없기 때문에 D가 가짜 이미지를 너무 잘 판별한다. 그래서 gradient 증가가 매우 적을 수 있다.
> 따라서, log(1-D(G(z))를 최소화하는 방면으로 학습했던 기존 방법 말고,
> **학습 초기**에는 logD(G(z))를 최대화하는 방면으로 학습한다. 그럼 초반에 gradient이 많이 증가해서 학습이 빨라진다.

- Early in learning: G is poor → D can reject samples with high confidence 
    - b/c they are clearly different from the training data
    - In this case, log(1-D(G(z))) saturates.
    - *minimize log(1-D(G(z)))* → **maximize logD(G(z))**
    - object function results in the same fixed point of the dynamics of G and D but *provides much stronger gradients early in learning.*

## Experiments

- `train data`: MNIST, TFD, CIFAR-10


- `generator`: mixture of rectifier linear activations, sigmoid activations


- `distriminator`: maxout activations
    - dropout applied in the training
    

- permit the use of dropout & other noise at intermediate layers of the generators
    - use noise as the input to only the `bottommost layer` of the `generator network`
    

## Advantages and Disadvantages

> 이 논문에서 제안한 GAN에도 장단점이 었으니.. **단점**이라고 함은 ① p_g(x)에 대한 명확한 식이 없는 것,
> ② D가 train하는 과정에서 G와 동기화가 되어야 한다는 것(D를 업데이트하지 않고 G를 train하지 말아야 함)
> 그리고 ③ learning step 동안 Boltzmann machine만큼의 negative chain이 반드시 있어야 한다는 점 정도 되겠다.

- Disadvantage
    - no explicit representation of p_g(x)
    - D must be synchronized well with G during training
        - G must not be trained too much without updating D → avoid "the Helvetica scenario"
            - G collapses too many values of z to the same value of x to have enough diversity to model p_data
    - much as the negative chains of a Boltzmann machine must be kept up to date between learning steps

> 하지만 **장점**이 있기에 이렇게 유명한 것 아닐까? 여기서 장점은 computional한 장점과 Adversarial Model을 사용함으로서 오는 장점 이렇게 두 개로 나뉠 수 있다.
>  ① gradient를 유지하기 위해 backpropagation만 써도 된다. 예전에는 Markov Chain이라는 것이 필요했는데, 이 GAN에서는
> 그것이 필요하지 않은 것이다. 또한 ② learning 와중에 interface가 필요하지 않다.
> ③ model을 설계하는 데에 다양한 기능이 포함될 수 있다.
> 
> **Adversarial Models**을 제안함으로서 얻는 statistical한 장점도 물론 있다. 
> ① generator network는 D를 통과한 gradient에 따라서만 업데이트된다.
> input이 G parameter로부터 바로 복사되지 않는다는 것을 뜻한다.
> ② degenerate distributions에서도 예리하게 재현될 수 있다. Markov Chain은 modes를 mix하기 위해 blurry한 distribution이 필요한 것과 반대임.
> 

- Advantages(primarily computional)
    - Markov chains are never needed → *only backprop* is used to obtain gradients
    - no interference is needed during learning
    - a wide variety of functions can be incorporated into the model
    

- Advantages(statistical): Adversarial Models
    - the generator network *updated only with gradients flowing through the discriminator.*
        - not being updated directly with data examples
        - this means that *components of the input are **not copied directly** into the generator's parameters*
    - can represent very sharp, even degenerate distributions
        - ↔ methods based on Markov chain: require that the distribution be somewhat blurry in order for the chain to be able to mix between modes.


## Conclusions

1. adding c as input to both G & D → obtain conditional generative model p(x|c)

2. approximate inference - x가 z를 예측하도록 network를 훈련시켜서 가능함. inference net이 generator가 train을 끝낸 후 fixed generator net에 의해 train될 수 있다는 이점 존재함.

3. 모든 conditionals p(xS | x6S) approximately model 가능. MP-DBM의 stochastic extension을 위해 adversarial net 사용 가능.

4. Semi-Supervised Learning: limited labeled data를 이용했을 때, generator나 inference net에서 추출된 feature들은 classifier의 성능 향상시킬 수 있음

5. 향수 G, D 조정 방향에 대해 더 나은 방향을 연구하거나 z sampling을 위한 더 나은 determining distributions 방법을 고안하면 성능 향상 가능

- Challenges in generative modeling

![image](https://user-images.githubusercontent.com/69420512/134119377-c2e98869-5985-4781-a1d8-4feddf75a49d.png)

- Figure 2
    - 기존의 gerative model들과 달리 model distribution의 actual sample를 보여 줌
        - hidden units의 conditional means가 아님
  Unlike most other visualizations of deep generative models, these images show actual samples from the model distributions, not conditional means given samples of hidden units. 
    - 그놈의 Markov Chain Mixing 안 사용해서 추출된 샘플들 사이 상관 관계 X