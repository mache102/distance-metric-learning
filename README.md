# contrastive_learning
Contrastive learning, contrastive loss, siamese networks

1. Contrastive learning

use pairs of inputs; model learns same and different inputs, their attrs
* self supervised

contrastive loss

A == A  
A != B 

imgs -> vec representations

## ---Defs---
1. gowrishankar  
$E = \frac{1}{2}yd^{2}+(1-y)max(\alpha-d, 0)$

2. jdhao
"Contrastive Loss is often used in image retrieval tasks to learn discriminative features for images. During training, an image pair is fed into the model with their ground truth relationship $y$: $y$ equals 1 if the two images are similar and 0 otherwise. The loss function for a single pair is:"  
$E = yd^{2}+(1-y)max(\alpha-d, 0)^{2}$

3. bekuzarov  
$L(W, (y, \overrightarrow{X_1}, \overrightarrow{X_2})^i) = yL_S(D^i_W)+(1-y)L_D(D^i_W)$

---
Where  
$y$ equals 1 if the two images are similar and 0 otherwise  
$d$ is the Euclidean distance between the image features $f_1$ and $f_2$, $d = ||f_1 - f_2||^{2}$  
$\alpha$ is the margin - Let us say if two images are similar, their distance should be greater than the margin. 


### Links
* https://jdhao.github.io/2017/03/13/some_loss_and_explanations/
contrastive loss explanation
* https://gowrishankar.info/blog/introduction-to-contrastive-loss-similarity-metric-as-an-objective-function/
contrastive loss metric
* https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246  
contrastive loss 3
* https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607  
contrastive learning explained
* https://towardsdatascience.com/siamese-networks-line-by-line-explanation-for-beginners-55b8be1d2fc6  
siamese network pairs, example
* https://keras.io/examples/vision/siamese_contrastive/  
keras siamese contrastive 

### Papers
* https://arxiv.org/pdf/2203.11075.pdf  
dense siamese
* https://arxiv.org/pdf/2207.01472.pdf  
Deep Contrastive One-Class Time Series Anomaly Detection  
**Notes:**   
representation & reconstructed representation = pos pair
* https://arxiv.org/pdf/2208.06616.pdf  
**Notes:**  
  - Img based CL methods may not work on time series data
    1. temporal dependencies (inefficiency)
    2. img augmentation not suited for time series
  - Solution: 
    1. **strong & weak TS augmentation**
    2. strong: tough cross-view prediction task in the next module; robust representation
    3. weak: small signal variations w/o affecting characteristics; minor chnges

## Implementation  
Summaries of sample contrastive learning implementations.  
Importing comes prior to the first step.  

Glossary:  
TVT = train/val/test  
[O] = optional
### 1. Siamese Network CL, CNN
- Steps:
  1. Load data; astype to proper; TVT split.
  2. [O] Visualize similar and different pairs.
  3. Build, compile, and fit a Siamese Network.
    * Implement contrastive loss during compilation.
  5. Plot result metrics.
