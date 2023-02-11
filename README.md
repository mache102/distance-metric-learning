# contrastive_learning
Contrastive learning, contrastive loss, siamese networks

1. Contrastive learning

use pairs of inputs; model learns same and different inputs, their attrs
* self supervised

contrastive loss

A == A  
A != B 

imgs -> vec representations

$E = \frac{1}{2}yd^{2}+(1-y)max(\alpha-d, 0)$

Where  
$d$ is the Euclidean distance between the image features  
$\alpha$ is the margin - Let us say if two images are similar, their distance should be greater than the margin.  
### Links
* https://gowrishankar.info/blog/introduction-to-contrastive-loss-similarity-metric-as-an-objective-function/
contrastive loss metric
* https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607  
contrastive learning explained
* https://towardsdatascience.com/siamese-networks-line-by-line-explanation-for-beginners-55b8be1d2fc6  
siamese network pairs, example
* https://keras.io/examples/vision/siamese_contrastive/  
keras siamese contrastive 

### Papers
* https://arxiv.org/pdf/2203.11075.pdf  
dense siamese
