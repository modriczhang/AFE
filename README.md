
## AFE

Personalized recommendation often relies on user historical behaviors to provide items for users. It is intuitive that future information
also contains essential messages as supplements to user historical behaviors. 

However, we cannot directly encode future information into models, since we are unable to get future information in online serving. In this work, we propose a novel adversarial future
encoding (AFE) framework to make full use of informative future features in different types of recommendation models. 

Specifically, AFE contains a future-aware discriminator and a generator. The future-aware discriminator takes both common features and future features as inputs, working as a recommendation prophet to judge
user-item pairs. In contrast, the generator is considered as a challenger, which generates items with only common features, aiming to confuse the future-aware prophet. The future-aware discriminator can inspire the generator (to be deployed online) to produce
better results. 

We further conduct a multi-factor optimization to enable a fast and stable model convergence via the direct learning and knowledge distillation losses. 

Moreover, we have adopted AFE on both a list-wise RL-based ranking model and a point-wise ranking model to verify its universality. 

In experiments, we conduct sufficient evaluations on two large-scale datasets, achieving significant improvements on both offline and online evaluations. 

Currently, we have deployed AFE on a real-world system, affecting millions of users.

### Requirements:
- Python 3.8
- Tensorflow 2.4.1

## Note

In the actual online system, AFE is a complex re-ranking framework implemented in C++. 
All models are trained based on a deeply customized version of distributed tensorflow supporting large-scale sparse features.

Without massive data and machine resources, training DRL-Rec is not realistic.

Therefore, the open source code here only implements a simplified version for interested researchers. If there are any errors, please contact me. Thanks!

## About

"A Peep into the Future: Adversarial Future Encoding in Recommendation" ([WSDM 2022](http://nlp.csai.tsinghua.edu.cn/~xrb/publications/WSDM-2022_AFE.pdf))
