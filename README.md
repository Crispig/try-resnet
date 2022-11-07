# Try resnet on CIFAR10



## This is a coursework on AI security

<!--Creating your own github account.Implementing your own deep neural network (in Pytorch, PaddlePaddle…).Training it on any datasets, for example, CIFAR10.Tuning a hyper-parameter and analyzing its effects on performance.Writing a README.md to report your findings.-->



## What I did

1. 参照resnet18来对CIFAR10数据集进行分类，使用pytorch手动搭建网络。
2. 对各部分设置随机种子，使结果可复现；寻找训练稳定的轮数，作为benchmark

3. 调整超参数，观察并分析模型性能。

4. 思考新的idea并实现，与benchmark对比并分析，改进idea的实现。



## 学习率对性能的影响

调整不同的学习率，观察准确率变化



## benchmark

batch size = 128

learning rate = 1e-3

epoch = 50

**ACC : 92.55%**



## Trying to innovate

过深的网络在反向传播时容易发生梯度弥散，ResNet中的shortcut相当于把以前处理过的信息直接再拿到现在一并处理，起到了减损的效果。

基于这样的设计理念，我尝试将这种shortcut从网络结构扩展到数据分布和序列分布之上。简单来说，我在resnet的基础上增加了更多的shortcut。

1. 在每个epoch中，存在多个batch的数据，每个batch会依次执行训练任务，我在每个block计算激活函数之前，将上一个batch训练时在该位置的输出shortcut到当前的输出。如下图所示

   ![image-20221107201411850](https://raw.githubusercontent.com/Crispig/Picbed/main/blog/img/20221107201420.png)

2. 在整个训练周期中，存在多个epoch，每个epoch会依次执行训练任务，我在每个block计算激活函数之前，将上一个epoch训练时每个batch在该位置的输出做一个平均之后，shortcut到当前的输出。如下图所示

   ![image-20221107201450496](https://raw.githubusercontent.com/Crispig/Picbed/main/blog/img/20221107201450.png)





### 结果

如果在训练一开始就启用shortcut

| method    | ACC        |
| --------- | ---------- |
| benchmark | **92.55%** |
| epoch_res | 10%        |
| batch_res | 88.36%     |



在训练45轮之后启用shortcut

| method    | ACC        |
| --------- | ---------- |
| benchmark | 92.55%     |
| epoch_res | **92.68%** |
| batch_res | 91.44%     |



### 结论

初步估计是因为如果一开始数据分布不稳定的时候就引入额外的信息会导致训练难以收敛，在准确率逐渐稳定之后，引入跨epoch的信息，会让模型获取更多信息。当然后续需要做更多的实验，来验证这种想法。