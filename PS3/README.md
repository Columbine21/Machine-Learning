# Generate Your Baby's Name

Date: 2020/5/29 

Author: Jason Yuan (Ziqi Yuan \<1564123490@qq.com\>)

## Quickly Start

-   Data 文件夹中包含了我们实验中使用的数据集。（readme.txt 声明了数据集出处&使用方法）

    -   female.txt 包含了 5000 + 条女性 参考姓名字符串
    -   male.txt 包含了约 3000  条男性 参考姓名字符串
    -   忽略 pet.txt 中的宠物姓名

-   EDA.ipynb 中我们对于数据集进行了简单的 nlp 方法的统计分析。了解了数据集的基本性质。

    -   读入了全部姓名字符串，展示了数据集的总条数。

    -   显示了数据集中male / female 中共同的 “中性姓名” （男女都有的姓名）

        <img src="/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS3/asset/EDA1.png" style="zoom:50%;" />

        -   发现了数据集中的一个小错误，在 famale.txt 中有两个连续的 `Gale` 人名
        -   并展示了  50 个 中性 的人名。

    -   显示了数据集中的一些特殊字符的出现。

        ![](/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS3/asset/EDA2.png)

        -   数据集中的 character_set 出了包含了 \[A-Z\] \[a-z\] 的英文字母。
        -   还包含了极少量 `'` `空格` `-` 三种特殊符号。
        -   然而，这三种符号出现次数过低，其中 `'` 出现一次，`空格` 出现两次， `-` 出现不足50次，远小于其余字母出现次数。
        -   故考虑到模型的性能，效率因素，在构造数据集的时候 filter 掉含有这三种特殊字符的姓名，并将姓名首字母转换为小写处理。

-   LanguageModel 文件夹保存了我们对于生成姓名的第一次基本尝试。

    -   模型虽然能够正确执行梯度下降方法。(使用 python nameGenerator.py 可以得到以下可视化内容)

        <img src="/Users/yuanziqi/Desktop/学习资料/大三下/机器学习/assignment/PS3/asset/attempt1.jpg" style="zoom:50%;" />

    -   该模型存在以下问题：

        ```python
        # 分别生成以 ‘a’ ‘g’ ‘j’ 开头的男性名字
        samples("male", all_categories, character_set, rnn, "agj")
        ------------------------------------------
        >>> arnnnn
        >>> geerrorrrrrr
        >>> jeeeeyeee
        
        # 分别生成以 ‘a’ ‘g’ ‘j’ 开头的女性名字
        samples("female", all_categories, character_set, rnn, "agj")
        ------------------------------------------
        >>> annnnll
        >>> geeriieeeeeae
        >>> jeaeeeeeeee
        ```

        -   模型由于其采用的数据 sampler 为随机从数据集中选择一个样本进行训练。可能由于每次的样本分布差异较大，导致模型每次迭代朝着	s最适合当前样本最优方向（与总体最优方向不一致）前进。
        -   模型可能存在 梯度爆炸 / 梯度消失 的问题，表现为我们的模型习惯于生成很长的名字，并且结果非常结巴。
            -   常见的解决方法是：使用 `grad_clip` 的方法进行 gradient 的限制（避免梯度爆炸）。
            -   常见的解决方法是：使用lstm的 gate 结构，避免梯度消失。

-   Generator 文件夹为实验的主要文件夹。（使用了padding 的 lstm 模型）
    -   在完成相应dependence的安装后，使用python main.py 进行模型训练和调试。