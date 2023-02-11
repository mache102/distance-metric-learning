
# triplet_loss 

https://gowrishankar.info/blog/introduction-to-contrastive-loss-similarity-metric-as-an-objective-function/
1. anchor
2. positive
3. negative

goal: maximize distance btwn difference of features btwn [anchor, pos] and [anchor, neg]  

### eq:  
$triplet loss = max(||f_a - f_p||^2 - ||f_a - f_n||^2 + m, 0)$  
$f_a$ = anchor features  
$f_p$ = positive features  
$f_n$ = negative features  

# cross_entropy_loss
https://jdhao.github.io/2017/03/13/some_loss_and_explanations/
* PT: Classification
* ||desired_prob_dist - pred_prob_dist||  (desired == true?)
### eq [k=2]:  
$cross entropy loss = -y\log(p)-(1-y)\log(1-p)$    
$y$ = sample label  
$p$ = pred. probability of sample belonging to class 1  
