# Controlled-Text-Generation-Image-Datasets
可控文本到图像生成数据集

## 2D 数据集

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

- COCO（COCO Captions） Dataset
  - 地址：https://cocodataset.org/#download
  - [简介](https://cocodataset.org/#home)：COCO Captions是一个字幕数据集，它以场景理解为目标，从日常生活场景中捕获图片数据，通过人工生成图片描述。该数据集包含330K个图文对。

- Visual Genome Dataset
  - 地址：http://visualgenome.org/
  - [简介](http://visualgenome.org/)：Visual Genome是李飞飞在2016年发布的大规模图片语义理解数据集，含图像和问答数据。标注密集，语义多样。该数据集包含5M个图文对。
    
- Conceptual Captions（CC） Dataset
  - 地址：https://ai.google.com/research/ConceptualCaptions/download
  - [简介](https://ai.google.com/research/ConceptualCaptions/download)：Conceptual Captions（CC）是一个非人工注释的多模态数据，包含图像URL以及字幕。对应的字幕描述是从网站的alt-text属性过滤而来。CC数据集因为数据量的不同分为CC3M（约330万对图文对）以及CC12M（约1200万对图文对）两个版本。

- YFCC100M Dataset
  - 地址：http://projects.dfki.uni-kl.de/yfcc100m/
  - [简介](http://projects.dfki.uni-kl.de/yfcc100m/)：YFCC100M数据库是2014年来基于雅虎Flickr的影像数据库。该库由一亿条产生于2004年至2014年间的多条媒体数据组成，其中包含了9920万张的照片数据以及80万条视频数据。YFCC100M数据集是在数据库的基础之上建立了一个文本数据文档，文档中每一行都是一条照片或视频的元数据。
 
- ALT200M Dataset
  - 地址：无
  - [简介]：ALT200M是微软团队为了研究缩放趋势在描述任务上的特点而构建的一个大规模图像-文本数据集。该数据集包含200M个图像-文本对。对应的文本描述是从网站的alt-text属性过滤而来。（私有数据集，无数据集链接）
    
- LAION-400M Dataset
  - 地址：https://laion.ai/blog/laion-400-open-dataset/
  - [简介](https://laion.ai/blog/laion-400-open-dataset/)：LAION-400M通过CommonCrwal获取2014-2021年网页中的文本和图片，然后使用CLIP过滤掉图像和文本嵌入相似度低于0.3的图文对，最终保留4亿个图像-文本对。然而，LAION-400M含有大量令人不适的图片，对文图生成任务影响较大。很多人用该数据集来生成色情图片，产生不好的影响。因此，更大更干净的数据集成为需求。

- LAION-5B Dataset
  - 地址：https://laion.ai/blog/laion-5b/
  - [简介](https://laion.ai/blog/laion-5b/)：LAION-5B是目前已知且开源的最大规模的多模态数据集。它通过CommonCrawl获取文本和图片，然后使用CLIP过滤掉图像和文本嵌入相似度低于0.28的图文对，最终保留下来50亿个图像-文本对。该数据集包含23.2亿的英文描述，22.6亿个100+其他语言以及12.7亿的未知语。



## 2. 文生图微调数据集
- DreamBooth：微调文本到图像的扩散模型以实现主题驱动的生成
  - 地址：https://github.com/google/dreambooth
  - 简介：该数据集包括 15 个不同类别的 30 个科目。其中 9 个是活体主体（狗和猫），21 个是物体。该数据集包含每个主题的可变数量的图像 (4-6)。
## 3. 可控文本生成图像数据集
- COCO-Stuff Dataset
  - 地址：https://github.com/nightrome/cocostuff
  - 简介：COCO-Stuff 使用像素级内容注释增强了流行的 COCO [2] 数据集的所有 164K 图像。这些注释可用于场景理解任务，例如语义分割、对象检测和图像字幕。
  - 命令行下载
  ```
  # Get this repo
  git clone https://github.com/nightrome/cocostuff.git
  cd cocostuff
  
  # Download everything
  wget --directory-prefix=downloads http://images.cocodataset.org/zips/train2017.zip
  wget --directory-prefix=downloads http://images.cocodataset.org/zips/val2017.zip
  wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
  
  # Unpack everything
  mkdir -p dataset/images
  mkdir -p dataset/annotations
  unzip downloads/train2017.zip -d dataset/images/
  unzip downloads/val2017.zip -d dataset/images/
  unzip downloads/stuffthingmaps_trainval2017.zip -d dataset/annotations/
  ```
-  Pick-a-Pic：用于文本到图像生成的用户首选项的开放数据集。
  - 地址：https://huggingface.co/datasets/yuvalkirstain/pickapic_v1
  - 简介：Pick-a-Pic 数据集是通过 Pick-a-Pic Web 应用程序收集的，包含超过 50 万个人类对模型生成图像的偏好示例。可以在此处找到带有 URL 而不是实际图像（这使其尺寸小得多）的数据集。
  - 命令行下载【国内加速】
  ```
  1. 下载hfd
  wget https://hf-mirror.com/hfd/hfd.sh
  chmod a+x hfd.sh
  2. 设置环境变量
  export HF_ENDPOINT=https://hf-mirror.com
  3.1 下载模型
  ./hfd.sh gpt2 --tool aria2c -x 4
  3.2 下载数据集
  ./hfd.sh yuvalkirstain/pickapic_v1 --dataset --tool aria2c -x 4
  ```


## 3D 数据集
## 1. 预训练数据集

## 2. 文生图微调数据集

## 3. 可控文本生成图像数据集
