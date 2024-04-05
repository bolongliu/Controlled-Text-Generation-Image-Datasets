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

- **COCO（COCO Captions） Dataset**
  - 地址：https://cocodataset.org/#download
  - [简介](https://cocodataset.org/#home)：COCO Captions是一个字幕数据集，它以场景理解为目标，从日常生活场景中捕获图片数据，通过人工生成图片描述。该数据集包含330K个图文对。
  - ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/bc65a58b-175f-4088-9323-c65cd2873064)

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

- Wikipedia-based Image Text (WIT) Dataset 基于维基百科的图像文本 (WIT) 数据集
  - 地址：https://github.com/google-research-datasets/wit/blob/main/DATA.md
  - [简介](https://github.com/google-research-datasets/wit/tree/main)：WIT（基于维基百科的图像文本）数据集是一个大型多模式多语言数据集，包含 3700 万多个图像文本集，其中包含 1100 万多个跨 100 多种语言的独特图像。我们以一组 10 个 tsv 文件（压缩）的形式提供 WIT。数据集总大小约为 25GB。这是整个训练数据集。如果您想快速开始，请选择任何一个大约 2.5GB 的文件，该文件将为您提供大约 10% 的数据，其中包含大约 350 万+ 图像文本示例集。
我们还包括验证集和测试集（各 5 个文件）。

- LAION-5B Dataset
  - 地址：https://laion.ai/blog/laion-5b/
  - [简介](https://laion.ai/blog/laion-5b/)：LAION-5B是目前已知且开源的最大规模的多模态数据集。它通过CommonCrawl获取文本和图片，然后使用CLIP过滤掉图像和文本嵌入相似度低于0.28的图文对，最终保留下来50亿个图像-文本对。该数据集包含23.2亿的英文描述，22.6亿个100+其他语言以及12.7亿的未知语。

- TaiSu(太素--亿级大规模中文视觉语言预训练数据集)
  - 地址：https://github.com/ksOAn6g5/TaiSu
  - [简介](https://github.com/ksOAn6g5/TaiSu)：TaiSu：166M大规模高质量中文视觉语言预训练数据集

- COYO-700M：大规模图像文本对数据集
  - 地址：https://github.com/kakaobrain/coyo-dataset/tree/main
  - [简介](https://github.com/kakaobrain/coyo-dataset/tree/main)：COYO-700M 是一个大型数据集，包含 747M 个图像文本对以及许多其他元属性，以提高训练各种模型的可用性。我们的数据集遵循与之前的视觉和语言数据集类似的策略，收集 HTML 文档中许多信息丰富的替代文本及其相关图像对。我们期望 COYO 用于训练流行的大规模基础模型，与其他类似数据集互补。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/a603c8f0-2795-41bf-aee8-bb31207171d2)




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

- DeepFashion-MultiModal
  - 地址：https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2
  - [简介](https://github.com/yumingj/DeepFashion-MultiModal?tab=readme-ov-file)：该数据集包括 15 个不同类别的 30 个科目。其中 9 个是活体主体（狗和猫），21 个是物体。该数据集包含每个主题的可变数量的图像 (4-6)。
  - 论文：[Text2Human: Text-Driven Controllable Human Image Generation](https://arxiv.org/pdf/2205.15996.pdf)
  - ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/2a57e76b-8495-466d-9d65-38f2abb7bd71)


- **DeepFashion**
  - 地址：https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
  - [简介](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)：该数据集包括 15 个不同类别的 30 个科目。其中 9 个是活体主体（狗和猫），21 个是物体。该数据集包含每个主题的可变数量的图像 (4-6)。
  - 论文：[ViscoNet: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet](https://arxiv.org/pdf/2312.03154.pdf)
 

https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
## 3D 数据集
## 1. 预训练数据集
- Multimodal3DIdent：从可控地面真实因素生成的图像/文本对的多模态数据集
  - 地址：https://zenodo.org/records/7678231
  - 简介：ICLR 2023 上发表的《多模态对比学习的可识别性结果》一文中介绍了用于生成 Multimodal3DIdent 数据集的官方代码。该数据集提供了可识别性基准，其中包含从可控地面真实因素生成的图像/文本对，其中一些在图像和文本模态之间共享，如以下示例所示。
  - 论文：[Identifiability Results for Multimodal Contrastive Learning](https://arxiv.org/pdf/2303.09166.pdf)
## 2. 文生图微调数据集

## 3. 可控文本生成图像数据集
