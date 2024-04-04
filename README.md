# Controlled-Text-Generation-Image-Datasets
可控文本到图像生成数据集

## 1. 预训练数据集
- Noah-Wukong Dataset
  - 地址：https://wukong-dataset.github.io/wukong-dataset/download.html
  - [简介](https://wukong-dataset.github.io/wukong-dataset/index.html)：Noah-Wukong 数据集是一个大规模多模态中文数据集。该数据集包含 1 亿个 <image, text> 对。

- Zero：微调文本到图像的扩散模型以实现主题驱动的生成
  - 地址：https://zero.so.com/download.html
  - [简介](https://zero.so.com/index.html)：Zero是一个大规模的中文跨模态基准，包含两个称为Zero-Corpus的预训练数据集和五个下游数据集。
    - **预训练数据集**
     2300 万个数据集（零语料库）。零语料库是从搜索引擎收集的，包含图像和相应的文本描述，是根据用户点击率从 50 亿个图文对中筛选出来的。
     230 万个数据集（Zero-Corpus-Sub）。零语料库的子数据集。在零语料库上训练 VLP 模型可能需要大量的 GPU 资源，因此还提供了包含 10% 图文对的子数据集用于研究目的。
    - **下游数据集**
       - ICM它是为图像文本匹配任务而设计的。它包含 400,000 个图像文本对，其中包括 200,000 个正例和 200,000 个负例。
       - IQM它也是一个用于图像文本匹配任务的数据集。与 ICM 不同，我们使用搜索查询而不是详细描述文本。同样，IQM 包含 200,000 个阳性病例和 200,000 个阴性病例。
       - ICR我们收集了 200,000 个图像-文本对。它包含图像到文本检索和文本到图像检索任务。
       - IQR IQR 也被提出用于图像文本检索任务。我们随机选择 200,000 个查询和相应的图像作为类似于 IQM 的带注释的图像-查询对。
       - Flickr30k-CNA我们聚集了专业的英汉语言学家，精心重新翻译Flickr30k的所有数据，并仔细检查每个句子。北京魔数数据科技有限公司为本数据集的翻译做出了贡献。
- Flickr 30k Dataset
  - 地址：https://shannon.cs.illinois.edu/DenotationGraph/data/index.html
  - [简介](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)：Flickr 30k 数据集包括从Flickr获取的图像。
 
  - 
## 2. 文生图微调数据集
- DreamBooth：微调文本到图像的扩散模型以实现主题驱动的生成
  - 地址：https://github.com/google/dreambooth
  - 简介：该数据集包括 15 个不同类别的 30 个科目。其中 9 个是活体主体（狗和猫），21 个是物体。该数据集包含每个主题的可变数量的图像 (4-6)。
## 3. 可控文本生成图像数据集
