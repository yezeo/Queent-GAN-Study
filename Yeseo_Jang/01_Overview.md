# 01. GAN Overview

## 1시간만에 GAN(Generative Adversarial Network) 완전 정복하기


- Supervised Learning
	- Discriminative Model
		- input > classification
		- how to classify


- Unspervised Learning
	- Generative Learning
		- Latent Code > training data(Image...)
		- learns the distribution of training data
		

- Probability Distribution
	- 확률 분포 >
	- 확률 변수 x의 차원을 늘려서 이미지로 > 이미지 분포 예측 가능할까?
	- 픽셀값


- 실제 밀도 함수에 대한 확률 밀도 함수(Probability density function)
	- 특정 특징 적게 등장 > 해당 픽셀에 해당하는 확률 밀도 값 작게 함
	- 특정 특징 많이 등장 > 해당 픽셀에 해당하는 확률 밀도 값 크기 함 > 데이터의 특성을 알 수 있음
		- 실재로 고차원 벡터이기 때문에 고차원에서 그림 상상하기
	- strange image 괴상한 이미지 > 이거에 해당하는 확률 분포 값은 낮음 (train 사진은 가지고 있지 않기 때문에)


GAN은 학습 데이터 분포와 근사한 데이터 분포를 갖는 모델을 만드는 것이 목표!!


### Discriminator, Generator

1. 진짜 이미지 가지고 학습
- Real Image > Discriminator


2. 가짜 이미지 가지고 학습
- 가짜를 가짜로 구별할 수 있도록 학습
- 고정된 벡터
- 진짜/가짜 binary하게 구별하면 되기 때문에 output은 0/1이 됨


- Generator
	- 값을 받아서 이미지를 생성해내야 함
	- 이미지를 가지고 Discriminator을 속여야 함
		- 결과로 1(sigmoid)이 나오도록(진짜 이미지로 판별하도록) 함
	- - Generator는 학습을 할수록 진짜같은 가짜 이미지를 만듦


- Discriminator
	- Objective Function을 최대화하는 것이 Discriminator의 목적
	- 진짜 data 분포로 sampling을 함


### PyTorch Implementation

github.com/yunjey/pytorch-tutorial

- Binary Cross Entropy Loss 사용
- Forward, Backward and Gradient Descent
	- Gradient값을 계산


### Non-Sturating Game

- Generator는 학습 초반에 형편없는 이미지를 만듦
	- Distriminator는 가짜라고 확신 → D(G(z)) = 0에 가까운 값 내보냄
		- 기울기의 절댓값이 작음 → logx를 최대화하는 쪽으로 함 → 기울기가 무한대
		- 최대한 빨리 벗어나려는 노력


### Theory in GAN

- 다시 보는 GAN의 목표: 실제 이미지와 만들어진 이미지의 확률 분포 차이를 줄이는 것
	- optimization을 시켰을 때 두 확률분포 간의 차이를 줄여 줌.

### 여러 GAN

- `DCGAN`
	- CNN으로 Discriminator, Deep Convolution으로 Generator
	- 기존에는 그냥 fc layer 썼다면 dis에서는 convolution, generator에서는 deep convolution, transfer convolution 이용
	- **fc 사용하지 않았다는 것 핵심**
		- fc 사용하면 de-fc 할 때 ㅠㅠ
	- fc layer, pooling layer 대신 strided conv 이용
	- batch norm 이용
	- adam optimizer(lr = 0.0002, betal = 0.5, beta2 = 0.999)
		- momentom이 2가지
		- 64x64 생성할 때 conv 4개 정도 쓸 때 실험적으로 결과 ㄱㅊ
	- Latent vector 가지고 **산술적 연산** 가능
		- man glasses - man without glasses + woman without glasses = woman with glasses


- `LSGAN`
	- 기존 Gan Loss는 Disc 속이기만 하면 됐음. 
	- Dicision Boundary(0.5 기준) 선
		- 파란색 선 근처에 있는 값이 좋은 값
		- 속였더라도 이상한 값 (핑크색) 가능 → 실재로 비슷한 이미지? 보장 불가
	- LSGAN에서는 3번째 그림처럼 분홍색 값을 끌어서 위로 올림
	- **sigmoid 없앰**
	- discriminator는 vanilla GAN과 똑같음
	- 기존에는 cross entropy loss → **least squares loss(L2)** 이용
		- 어떤 값이 나오든 dicision boundary 쪽으로 몰리게 함
		- 분홍색 점의 경우 높은 값을 내보냈지만 L2 쓰면 dicision boundary 쪽으로 올려져서 이미지 퀄리티 좋아짐
	- DCGAN 결과보다 더 좋아짐


- `SGAN`(Semi-Supervised GAN)
	- Discriminator가 진짜/가짜 구분 말고 class 구분을 함
		- 가짜 이미지 생성할 때 generator에 latent vector + one-hot vector 넣어 줌
		- 가짜 이미지 2를 생성해라 → disciminator는 fake class 구분, generator는 input으로 준 one-hot vector와 output 똑같이 나오도록 함
	- training with real images → supervised learning
	- training with fake images → unsupervised learning
	- e.g., 각각의 character의 포즈 class one hot vector 
		- z 벡터 똑같고 class 다르게 주면 캐릭터는 똑같고 포즈(class)가 변하게 됨
	- 무엇을 생성할지 **class**를 줄 수 있음.
	- discriminator가 내뱉을 수 있는 output vector: class 개수 + 1 (real + fake 1개)
		- 가짜가 들어오면 class 구분 안 하고 그냥 **가짜라고 구분**
		- 한계점: generator가 학습할수록 *비슷한* 실제 이미지 만듦


- `ACGAN` (Auxiliary Classifier GAN)
	- discriminator가 해야 되는 task 2개
		- 1. real or fake 구별
		- 2. one-hot vector representing - 진짜든 가짜든 class 구별
			- output vector의 개수: class 개수
	- 마지막 단에 있는 fc layer만 하나는 sigmoid / 하나는 softmax로 독립적으로 이용
	- 2가지 task 동시에 잘해야 함, multi-task learning이 됨
	- fake image train할 때
		- latent vector z, one-hot vector representing- 넣어 줌
		- semi-supervised learning과 같이 class 정보 넣어 줌
		- **가짜임에도 불구하고 class를 구별해 줘야 함** → 더 좋은 이미지 만들어 줄 수 있음
			- generator에서 data augmentation 효과 받을 수 O

### Extensions of GAN

- `CycleGAN`: Unpaired Image-to-Imaged Translation
	- image의 domain/style을 바꾸도록 함
	- paired image 데이터가 없이(parallel image) 없이 un-supervised learning을 해도 가능하도록 (**in the absence of paired examples**)
	- Latent 받지 않고 image를 받게 됨
	- Discriminator
		- 목표: A domain image → B domain image
		- B domain image만 줌 
		- generator는 A를 받았을 때 B를 내보내게 됨 (B만 참이라고 판단)
			- 포즈가 바뀐다든지 해도 B domain 가지면 discriminator 속일 수 있음 → 이걸 방지해야 함
			- 모양을 유지해 주는 것도 task의 목표임
	- 얼룩말 이미지 → 말 이미지 → 다시 얼룩말 이미지


- `StackGAN`: Text to Photo-realisctic Image Synthesis
	- text 주고 text에 해당하는 이미지를 만듦
	- G1으로 저해상도 이미지 생성 → 받아서 upscaling하게끔 또 다른 generator G2
		- 이미지 size도 2개, generator도 2개 둠
	- 그냥 G 한 개로 했을 때는 random한 벡터 z를 주고 바로 가짜 이미지 생성. 좀 어려움
		 - G1, G2로 나누었을 때 더 잘됨


- 다른 task들
	- Visual Attribute Transfer
	- User-Interactive Image Colorization