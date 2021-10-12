# 05. Conditional Image Synthesis With Auxiliary Classifier GANs

- new methods for the improved training of GANs for **`image synthesis`**

-  expand on previous work for image quality assessment to provide two new analyses 
   - for **assessing the discriminability** and **diversity of samples from class-conditional image synthesis models**

- analyses demonstrate
    - **high resolution samples provide class information** not present in low resolution samples.

- 128x128 samples가 다른 더 작은 resized sample들보다 더 discriminable했다
- 그리고 84.7%의 클래스는 실제 데티어와 유사한 다양성을 보이는 샘플을 가진다

## I. Introduction

- Demonstrate an image synthesis model for all 1000 ImageNet classes at a 128x128 spatial resolution (or any spatial resolution - see Section 3).
- Measure how much an image synthesis model actually uses its output resolution (Section 4.1).
- Measure **perceptual variability and ’collapsing’ behavior in a GAN with a fast**, easy-to-compute metric (Section 4.2).
- Highlight that **a high number of classes is what makes ImageNet synthesis difficult for GANs and provide an explicit solution** (Section 4.6).
- Demonstrate experimentally that GANs that **perform well perceptually are not those that memorize a small number of examples** (Section 4.3).
- Achieve state of the art on the **Inception score metric** when trained on CIFAR-10 without using any of the techniques from (Salimans et al., 2016) (Section 4.4).

---

## II. Background

- GAN: consists of two neural networks trained in *opposition* to one another
    - The Generator G: takes as input
        - a random noise vector *z*
        - outputs an image *X_fake = G(z)*
    - The Discriminator D: receives as input either 
        - a training image or a synthesized image from the generator
        - outputs a probability distribution *P(S|X) = D(X)* over possible image sources
    - The Discriminator is trained to **maximize the log-likelihood** it assigns to the correct source:
    - The Generator is trained to **minimize the second term**
    
![image](https://user-images.githubusercontent.com/69420512/136829552-e03d2ce5-e062-44ce-8c2d-eba68fcebc7a.png)



---

## III. AC-GANs

> AC-GAN에서, 모든 generated된 셈플은 class label과 noise를 가지고 있다. 이때 G는 generate images에 둘 다 이용하고,
> D는 sources와 class labels 둘 다에 probability distribution을 준다. 
> 그리고 objetive function는 Ls, Lc를 모두 가지는데
> 이제 D는 Ls+Lc를 최대화하는 방향으로, G는 Lc-Ls를 최대화하는 방향으로 작동한다.
> 이때 AC-GANs의 noise인 z는 class label에 독립적임.

- Every generated sample has a 
    - corresponding class label `c~p_c`
    - noise `z`


- **G**: uses both to generate images *X_fake = G(c, z)*

- **D**: gives both a probability distribution over sources and a probability distribution over the class labels.
    - P(S|X), P(C|X) = D(X)


- **The Objective Function**
    - 1) **`Ls`**: the log-likelihood of the **correct source**
    - 2) **`Lc`**: the log-likelihood of the **correct class**
    
![image](https://user-images.githubusercontent.com/69420512/136820760-11e1f220-0a12-41fa-87b8-3dfcaead65aa.png)


- **D**: trained to **maximize** `Ls + Lc`

- **G**: trained to **maximize** `Lc - Ls`

- AC-GANs learn a representation for *z* that is **independent of class label**

> 구조적으로, 기본적으로 존재하는 모델과 크게 다르지 않으면서도 AC-GAN은 우수한 결과를 생성하고 training을 안정화(stabilize)시킨다.
> 
> 특히 AC-GAN은 1) output resoultion(출력 해상도)를 model이 사용하는 정도를 측정하는 방법, 2) model로부터 생성된 sample의 perceptual variability를 측정하는 방법, 3) 1000 ImageNet 클래스로부터 
> 128x128 샘플을 생성한 experimental analysis(실험적 분석) 때문에 더 technical contributions가 되었다고 한다.

- Structurally, not tremendously different from existing models
    - can modification standard GAN: **produces excellent results and appears to stabilize training**
    - AC-GAN model to be only part of the **technical contributions**
        - `methods` for measuring **the extent to which a model makes use of its given output resolution**
        - `methods` for measuring **perceptual variability of samples from the model**
        - a thorough **experimental analysis of a generative model of images** that creates 128x128 samples from all 1000 ImageNet classes

> 이전 연구에서는 fixed된 model에서 train될 때, class의 갯수를 늘리는 것은 model output의 결과의 질을 감소시킨다고 했다.
> 
> 그런데 ACGAN은 큰 dataset을 subset들로 작게 나누어도 된다. 그리고 각각의 subset들로 G와 D를 training한다.
> ImageNet 실험도 이제 100개의 AC-GANs를 ensemble했고, 각각은 10개의 class로 split된 것이었다.
> 

- Early experiments demonstrated: 
    - `increasing the number of classes` trained on while holding the model fixed `decreased` the quality of the model outputs

- AC-GANs permits **separating large datasets into `subsets` by class** and training a generator and discriminator for *each subsets.*
    - All ImageNet experiments are conducted using an **ensemble** of 100 AC-GANs
    - each trained on a 10 class split

---

## IV. Results

- generator G: series of `deconvolution` layers
    - transform the noise *z* & class *c* into an image
    

- We train two variants of the model architecture for generating images at
    - 128x128 spatial resolutions
    - 64x64 spatial resolutions
    

- discriminator D: dep convolutional neural network
    - with Leaky ReLU nonlinearity
    

- `Evaluating`
    - measure the quality of the AC-GAN by building several *ad-hoc* measures for image sample discriminability and diversity.
    

- `Hope`
    - provide quantitative measures that may be used to *aid training* and *subsequent development* of image synthesis models.
    
### IV-1. Generating High Resolution Images Improves Discriminability

- Goal: To show that **synthesizing higher resolution images leads to increased discriminability**

- 연구 결과, 128x128 model이 64x64, 32x32 model보다 accuracy가 더 높게 나왔다.
    - 또한 64x64 resolution model이 64 spatial resoultion에서 128x128 model보다 **discriminability**가 적다는 결과를 알 수 있었다.
  
  
- `의의`: *first* that attempts to measure the extent to which an image synthesis model is 'making use of its given output resolution'
    - the first work to consider the issue at all
    
- can be applied to any image synthesis model
    - a measure of 'sample quality' can be constructed
    - can be applied to any type of synthesis model
        - as long as there is an easily computable notion of sample quality and some method for 'reducing resolution'
    - audio synthesis
    
    
### IV-2. Measuring the Diversity of Generated Images

- well-known failure model of GANs 
    - the generator will collapse and output a single prototype that maximally fools the discriminator
    
- We seek a **complementary metric to explicitly evaluate the intra-class perceptual diversity** of samples generated by the AC-GAN
    - A class-conditional model of images is not very interesting if it only outputs one image per class


- The most successful of these is multi-scale structural similarity(MS-SSIM)
    - MS-SSIM: multi-scale variant of a well-characterized perceptual similarity metric tat attempts to discount aspects of an image that are not important for human perception
    - 0에서 1까지의 범위 안에서 표현되고, 높을수록 similar image라는 것을 뜻한다.
    

- MS-SSIM을 diversity의 관점에서 보면, **lower mean MS-SSIM scores를 가질수록, diversity는 high하다.**


- `Fig 5`: Classes for which the generator *'collapses'* will have increasing mean MS-SSIM scores.

### IV-3. Generated Images are both Diverse and Discriminable

- examine how these metrics *interact*?

- GANs: **drop modes are most likely to produce low quality images**
    - contrast: that they achieve high sample quality at the expense of variability

### IV-4. Comparison to Previous Results

- `Log-likelihood`: a coarse and potentially inaccurate measure of sample quality

- provides additional evidence that AC-GANs are effective even *without the benefit of class splitting*
    - `fig 7`: AC-GAN samples posses global coherence absent from the samples of the earlier model

### IV-5. Searching for Signatures of Overfitting

- must be investigated: AC-GAN has *overfit on the training data*


- first check of *the network **does not memorize** the training data*
    - identify the nearest neighbors of image samples by L1 distance in pixel space
    - result: **do not resemble the corresponding samples**
    - this provides evidence that the AC-GAN is not **merely memorizing** the training data
    

- **overfitting**
    - the generator learned that certain combinations of dimensions correspond to semantically meaningful features
    - no discrete transitions or 'holes' in the latent space
    - AC-GAN to exploit the structure of the model
        - AC-GAN with z fixed but altering the class label corresponds to generating samples with the same 'style' across multiple classes
        - Elements of the same row have the same *z*.
    - Although the class changes for each column, elements of the global structure are preserved
        - **indicating that AC-GAN can represent certain types of `compositionality`**
    
### IV-6. Measuring the Effect of Class Splits on Image Sample Quality

-  final model: divide 1000 ImageNet classes across 100 ACGAN models.
    - class의 diversity를 줄이는 것의 이점
    - split의 장점 2가지
        - 1. the number of classes per split
        - 2. intra-split diversity
    - **training a fixed model on more classes** harms the model’s ability to produce compelling samples.
        - Performance *on larger splits can be improved by giving the model more parameters.*


그런데 small split으로는 **sufficient하지 않다!!**
good performance를 위해서는 더 많은 것이 필요하다.

We were unable to find conclusive evidence that the selection of classes in a split significantly affects sample quality.


### IV-7. Samples from all 1000 ImageNet Classes

- generate 10 samples from each of the 100 ImageNet classes
    - no other image synthesis work has included a similar analysis

---

## V. Discussion

- introduced the `AC-GAN` architecture
    - demonstrated that AC_GANs can generate globally coherent ImageNet samples
    

- provided a new quantitative **metric** for image discriminability
    - as a function of *spatial resolution*
    - using -> demonstrated our samples are more *discriminable* and *performs a naive resize operation*
    

- anlysis the *diversity* of our samples with respect to the training data and provided some evidence
    - that the image samples from the majority of class: *comparable in diversity to ImageNet data*
    

- `Much work needs...`
    - improve the *visual discriminability*
    - augment the discriminator with a pre-trained model to *perform additional supervised tasks*
    

- `Improving the reliability of GAN training`
    - AC-GAN model can perform **semi-supervised learning**
      - by **ignoring the component of the loss arising from class labels** when a label is unavailable for a given training image.