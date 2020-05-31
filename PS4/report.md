# 机器学习（四）ML-GCN: 基于图卷积网络多标签图像分类

Date: 2020/5/30

Author: Ziqi Yuan \<Columbine21@1564123490@qq.com>

题目重述：描述一个神经网络模型的应用例子，包括问题的描述，输入输出数据，训练测试数据集，深度网络哦模型结构，损失函数，训练方法，以及其他需要说明的内容。

## （0）问题描述

随着科学技术的进步与发展，图像作为信息传播的重要媒介，在通信、无人驾驶、医学影像分析、航天、遥感等多个领域得到了广泛的研究，并在国民社会、经济生活中承担着更加重要的角色。

在图像分类问题中，根据分类任务的目标不同，可以将图像分类任务划分成两部分:（1）单标签图像分类；（2）多标签图像分类。现实生活中的图片中往往包含多个类别的物体，这也更加符合人的认知习惯。多标签图像分类可以告知我们图像中是否同时包含这些内容，这也能够更好地解决实际生活中的问题。

-   问题分类：多标签图像分类（识别出一幅图像中所有的事物）
-   模型创新点：模型使用 图卷积神经网络（GCN）从 label 的先验表示，捕捉各个 label 之间的相互依赖关系，进而提升模型性能。同时提出了创新的 权重矩阵 作为关联矩阵以解决 GCN 模型可能导致的 over-smoothing 问题。

问题举例：我们给出此类问题的一个简单的例子如下：

<img src="/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS4/asset/e1.png" style="zoom:50%;" />

模型以 左一图片 作为输入，模型期望在`Person Sports` ,` Ball Tennis` ,`Racket ` 三个 label 的位置输出为 1 ，表示含有该 object， 而在其他位置输出为 0，表示图像中不含有其他 object。   

## （1）训练/测试 数据集

-   模型分别使用在图像任务上常见的两个数据集：
    -   MS-COCO ： Microsoft COCO 是在图像识别的广泛使用的基准数据集。 它包含82,081张图像作为训练集和40,504张图像作为验证集。 这些对象分为80类，每个图像约有2.9个对象标签。 
    -    VOC 2007 ：PASCAL Visual Object Classes Challenge (VOC 2007) 是另一个用于多标签识别的流行数据集。 它包含来自20个对象类别的9,963张图像，分为训练，验证值和测试集。

## （2）模型结构

###     （2.1）模型概述

模型总体结构如下图所示：（以下用 $C$ 表示模型预测的不同标签的数量）

![](/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS4/asset/m1.png)

-   模型输入：经过 transform 的三通道图像数据 （[batch_size，3，W，H]）
-   模型输出；每个类别出现在图片中的概率向量 （[batch_size, C]）

如模型结构图上半部分所示，模型使用预训练的 ResNet101捕获图像特征（将输入的 $3$ 通道图像[batch_size，3，W，H] 转化为 $D$ 通道的特征图 [batch_size，D，W，H]），模型使用然后全局最大池化层得到每个通道的输出特征图的最大值。（将 $D$ 通道的特征图 [batch_size，D，W，H] 转化为  各个特征图最大值 [batch_size，D，1]）

如模型结构图下半部分所示，模型使用 GCN（用各个 label 作为图的节点，使用 **全局共现矩阵** 构建边）来建模标签之间的相互依赖性。 （将输入标签的初始单词嵌入 [C，d] 转化为包含label之间依赖关系的标签嵌入 [C，D]）。

接下来将对于模型的核心部分结合实现代码做详细说明：

###     （2.2）图像特征提取器 ResNet101 网络

模型接受经过 transform 的图像数据，各个图像经过 transform 均变成 448 * 448 像素的三通道图像。进过预训练的 ResNet101 网络的 "conv5-x" layer 后得到 2048 * 14 * 14 的特征图矩阵。（在附录中，我们详细的介绍说明集成在torchversion.model里面的ResNet 的代码构架，并详细推倒 ResNet101 网络如何从 448 * 448 * 3 的输入得到 2048 * 14 * 14 的特征图矩阵）然后使用Global Max Pooling 得到每个特征图的最大值，作为图像特征。

​                                                $x = f_{GMP}(f_{cnn}(I;\theta_{cnn})) \in R^D$ , $D=2048$ 

<img src="/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS4/asset/m2.png" style="zoom:50%;" />

核心代码展示如下：

```python
self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
self.num_classes = num_classes
self.pooling = nn.MaxPool2d(14, 14)
```

###     （2.3）图卷积神经网络 GCN

**GCN简介：**

​	图卷积神经网络 GCN 在半监督分类任务中提出，用以通过节点之间信息传递更新节点表示。模型以 原始节点表示矩阵 $H^l\in R^{n\times d}$ 和图的关联矩阵 $A\in R^{n\times n}$ 作为输入。（$n$ 表示节点个数，$d$ 表示节点特征维度）

​	空域图卷集引入了以下图上的卷积运算：$H^{l+1} = h(\hat{A}H^lW^l)$ , $W^l$ 是待学习的转化矩阵，$\hat{A}$ 是标准化的关联矩阵，$h$ 表示非线性变换（本模型中采用 LeakyReLU 作为非线性函数）

<img src="/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS4/asset/m3.png" style="zoom:50%;" />

有了以上理论基础，我们希望通过模型实现代码进一步了解 GCN 模型的构建方法，下给出单层GCN实现方法：

```python
class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907"""
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 使用 Parameter 构造模型的参数（默认加入model.parameters()）
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
      """参数初始化 weight/bias 参数均使用 uniform 初始化"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 模型前向传播可以简单理解为两次矩阵乘法，第一个矩阵乘法将模型输入特征进行线形变换，第二次矩阵乘法根据关联矩阵对于变换后特征进行加权求和。
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
```

**基于 GCN 模型的分类器**

​	我们的模型使用了两层 stack GCN 第一层 GCN 以 label 的词向量（无特殊说明，指Glove词向量 300dim）作为输入，输出1024维 label 中间表示，作为第二层 GCN 的输入，第二层 GCN 输出 2048 维 label 表示 $W \in R^{C \times 2048}$ 作为label 的最终表示方法。与第一部分图像特征 $x \in R^{2048\times 1}$ 进行向量乘法运算，得到 $\hat{y} = W\cdot x \in R^{C}$ 作为每个类别的概率，计算损失函数。（使用的损失函数见**第三节训练方法**） 

**GCN 中关联矩阵的构造**

-   记 $P(L_i|L_j)$ 表示在 $L_j$ 标签出现的条件下，$L_i$ 标签也出现的条件概率。

-   作者首先考虑了一种基本的方法：首先计算各个 label 之间的 $P(L_i|L_j)$ 然后使用 threshold ($\tau$) 对概率进行 0/1 划分。即当 $P(L_i|L_j) > \tau$ 则矩阵相应位置为 1，反之为 0。

-   然而直接入上述方法进行关联矩阵构造会导致 over-smoothing 的问题。作者进一步提出了 re-weighted 策略，根据如下的方法：
    $$
    A'_{ij} = \left\{
    \begin{aligned}
    \frac{p}{\sum_{j=1 \and i\neq j}^CA_{ij}} &  & i \neq j \\
    1-p & & i = j
    \end{aligned}
    \right.
    $$
    可以看出 $p$ 越大，节点越趋向于保留自己本身的特征。我们下面结合实现代码进行分析。

    数据集中给出了coco_adj.pkl 文件，其中包含了每个 label 出现的总次数 'nums' (80,) 和 'adj' 就是共现矩阵 $M$ (80 * 80)

    <img src="/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS4/asset/coco_adj.png" style="zoom:50%;" />

    我们接下来看实现代码：

    ```python
    def gen_A(num_classes, t, adj_file):
        import pickle
        result = pickle.load(open(adj_file, 'rb'))
        _adj = result['adj']
        _nums = result['nums']
        # first get the conditional probability _nums[i,j] = P(xj | xi).
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums
        # clip with threshold.
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1
        # 到上面为止，基础的方法已经完成，下面操作是re-weight策略
        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        # 对角元素值加一
        _adj = _adj + np.identity(num_classes, np.int)
        return _adj
    
    def gen_adj(A):
        # Receive the output of gen_A as input.
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj
    ```

## （3）训练方法

-   损失函数：使用传统多分类损失函数：
    $$
    L = \sum_{c=1}^C y^c\log(\sigma(\hat{y}^c)) +(1-y^c)\log(1-\sigma(\hat{y}^c))
    $$
    

-   优化器：使用 SGD 优化器，设置 momentum 为 0.9 ，Weight decay 为 $1e-4$ ,初始学习率为 0.01, 每40 epochs 学习率衰减 到原先的 0.1 倍。网络一共运行 100 epochs。

-   coding：

    ```python
    def get_config_optim(self, lr, lrp):
        return [
        {'params': self.features.parameters(), 'lr': lr * lrp},
        {'params': self.gc1.parameters(), 'lr': lr},
        {'params': self.gc2.parameters(), 'lr': lr},
        ]
    
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    ```

## （4）评价标准

该论文实验中采用了很多中不同的评价方式，在两个数据集上组了充分的评测/案例分析等。论文中提到的几种核心评价标准列举如下：

-   average overall F1 (OF1)
-   average per-class F1 (CF1) 
-   mean average precision (mAP)



至此，我们的模型分析大致上画上了句号，在附录中，我们期望通过阅读torchvision.models source code 分析ResNet101 进行特征提取的流程。

## （5）Appendix 

### ResNet101 源码分析

文件夹位置：torchvision.models.resnet.py

从（2.2）部分的代码中，我们明白我们模型使用了Resnet 的第一层 CNN 网络（包含conv1,bn1,relu, maxpool）以及后续的 layer1-layer4。我们希望能够从源代码中得到这些层的详细信息。我们进入源代码部分。

首先论文实现源代码通过 `models.resnet101(pretrained=pretrained)` 进入 `torchvision.models.resnet.py` 

```python
def resnet101(pretrained=False, progress=True, **kwargs):
    """
    可以看到，我们在最顶层暴露给用户的接口只有是否 pretrain 这一个常用的参数
    同时，模型使用Bottleneck 的网络结构，通过调用 _resnet 构造了模型。
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
  
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # 函数通过调用 ResNet 完成对于模型的构造，我们分析的核心在于这个ResNet结构。
    model = ResNet(block, layers, **kwargs)
    # 这里完成了模型参数加载的功能，
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

通过上面的分析，我们了解到，我们ResNet101的核心结构应该参考 `ResNet(Bottlenect, [3, 4, 23, 3])` 。

下主要分析 ResNet 类究竟干了什么。我们代码中忽略不常用的 cnn 参数（dilation/group等）和前向传播过程，仅从核心模型结构的角度出发，研究模型的结构。幸运的是，我们这里看到了我们希望看到的 `conv1,bn1,relu, maxpool` 以及 `layer1-layer4` 结构。

```python
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
```

-   第一层 CNN 结构分析：

    -   self.conv1:  nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        -   输入：[batch_size, 3, 488, 488] 输出 : [batch_size, 64, 224, 224]
    -   self.bn1: nn.BatchNorm2d(self.inplanes)
    -   self.relu = nn.ReLU(inplace=True)
    -   self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        -   输入：[batch_size, 64, 244, 244] 输出：[batch_size, 64, 112, 112]

-   layer1-layer4 结构分析：可以看到 layer1-layer4 分别由 [3, 4, 23, 3] 个 block 结构构成。

    -   且只有在layer2-layer4 中的第一个 block 结构中传入了 stride=2 的参数将特征图缩小为1/2。故此输出特征图尺寸应为 112 / 8 = 14 符合我们的预期。

    -   每个 block 结构如下所示

        ```python
        class Bottleneck(nn.Module):
            expansion = 4
            __constants__ = ['downsample']
        
            def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                         base_width=64, dilation=1, norm_layer=None):
                super(Bottleneck, self).__init__()
                if norm_layer is None:
                    norm_layer = nn.BatchNorm2d
                width = int(planes * (base_width / 64.)) * groups
                
                self.conv1 = conv1x1(inplanes, width)
                self.bn1 = norm_layer(width)
                self.conv2 = conv3x3(width, width, stride, groups, dilation)
                self.bn2 = norm_layer(width)
                self.conv3 = conv1x1(width, planes * self.expansion)
                self.bn3 = norm_layer(planes * self.expansion)
                self.relu = nn.ReLU(inplace=True)
                
             def forward(self, x):
                ...
                # here is the core idea of "Res", 残差结构
                out += identity
                out = self.relu(out)
                return out
        ```

        可以看到，每个block 结构都前后使用了 1x1 卷积对特征图的 channel 进行变换，极大程度上减小了3x3卷积操作所需要的参数个数。并且保持了图像尺寸不变（stride=1）的情况。最终的输出 channel 为 expansion * planes。

    -   这样我们 `self.layer4 = self._make_layer(block, 512, layers[3], stride=2)` 的输出便是 4 * 512 = 2048 亦符合我们的预期。

    这样，我们对于ResNet101 的分析也就可以结束了。

### 参考资料

论文参考地址：[https://arxiv.org/abs/1904.03582](https://arxiv.org/abs/1904.03582) (CVPR 2019)

参考代码地址：[https://github.com/Megvii-Nanjing/ML-GCN](https://github.com/Megvii-Nanjing/ML-GCN)