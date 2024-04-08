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
  - 地址：https://huggingface.co/datasets/kakaobrain/coyo-700m
  - [简介](https://github.com/kakaobrain/coyo-dataset/tree/main)：COYO-700M 是一个大型数据集，包含 747M 个图像文本对以及许多其他元属性，以提高训练各种模型的可用性。我们的数据集遵循与之前的视觉和语言数据集类似的策略，收集 HTML 文档中许多信息丰富的替代文本及其相关图像对。我们期望 COYO 用于训练流行的大规模基础模型，与其他类似数据集互补。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/a603c8f0-2795-41bf-aee8-bb31207171d2)

- WIT：基于维基百科的图像文本数据集
  - 地址：https://github.com/google-research-datasets/wit
  - [简介](https://github.com/google-research-datasets/wit)：基于维基百科的图像文本（WIT）数据集是一个大型多模态多语言数据集。 WIT 由一组精选的 3760 万个实体丰富的图像文本示例组成，其中包含 1150 万个跨 108 种维基百科语言的独特图像。其大小使得 WIT 能够用作多模式机器学习模型的预训练数据集。
  - 论文 [WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning](https://arxiv.org/pdf/2103.01913.pdf)
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/77851e32-1153-488e-99c7-39c154a126d6)


- DiffusionDB
  - 地址：https://huggingface.co/datasets/poloclub/diffusiondb
  - [简介](https://github.com/poloclub/diffusiondb)：DiffusionDB 是第一个大规模文本到图像提示数据集。它包含由稳定扩散使用真实用户指定的提示和超参数生成的 1400 万张图像。这个人类驱动的数据集前所未有的规模和多样性为理解提示和生成模型之间的相互作用、检测深度伪造以及设计人机交互工具以帮助用户更轻松地使用这些模型提供了令人兴奋的研究机会。DiffusionDB 2M 中的 200 万张图像被分为 2,000 个文件夹，其中每个文件夹包含 1,000 个图像和一个 JSON 文件，该文件将这 1,000 个图像链接到它们的提示和超参数。同样，DiffusionDB Large 中的 1400 万张图像被分为 14000 个文件夹。
  - 论文 [DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models](https://arxiv.org/pdf/2210.14896.pdf)
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/9eb37d8c-be99-4443-af43-8bcae8dd0584)





## 2. 文生图微调数据集
- DreamBooth：微调文本到图像的扩散模型以实现主题驱动的生成
  - 地址：https://github.com/google/dreambooth
  - 简介：该数据集包括 15 个不同类别的 30 个科目。其中 9 个是活体主体（狗和猫），21 个是物体。该数据集包含每个主题的可变数量的图像 (4-6)。
## 3. 可控文本生成图像数据集
- **COCO-Stuff** Dataset
  - 地址：https://github.com/nightrome/cocostuff
  - 简介：COCO-Stuff 使用像素级内容注释增强了流行的 COCO [2] 数据集的所有 164K 图像。这些注释可用于场景理解任务，例如语义分割、对象检测和图像字幕。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/6073dce1-977f-45b4-99ca-dbc51b8fa8df)
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
-  ***Pick-a-Pic：用于文本到图像生成的用户首选项的开放数据集**
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

- **DeepFashion-MultiModal**
  - 地址：https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2
  - [简介](https://github.com/yumingj/DeepFashion-MultiModal?tab=readme-ov-file)：该数据集是一个具有丰富多模态注释的大规模高质量人体数据集。它具有以下属性：它包含44,096张高分辨率人体图像，其中12,701张全身人体图像。对于每张全身图像，我们手动注释 24 个类别的人体解析标签。对于每张全身图像，我们手动注释关键点。每张图像都手动标注了衣服形状和纹理的属性。我们为每张图像提供文字描述。DeepFashion-MultiModal 可应用于文本驱动的人体图像生成、文本引导的人体图像操作、骨架引导的人体图像生成、人体姿势估计、人体图像字幕、人体图像的多模态学习、人体属性识别、人体解析预测等,该数据集是在 Text2Human 中提出的。
  - 论文：[Text2Human: Text-Driven Controllable Human Image Generation](https://arxiv.org/pdf/2205.15996.pdf)
  - ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/2a57e76b-8495-466d-9d65-38f2abb7bd71)


- **DeepFashion**
  - 地址：https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
  - [简介](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)：该数据集是一个大规模的服装数据库，它有几个吸引人的特性：首先，DeepFashion包含超过80万张多样化的时尚图片，从摆好姿势的商店图片到不受约束的消费者照片，构成了最大的视觉时尚分析数据库。
其次，DeepFashion标注了丰富的服装单品信息。该数据集中的每张图像都标有 50 个类别、1,000 个描述性属性、边界框和服装地标。第三，DeepFashion 包含超过 300,000 个跨姿势/跨域图像对。使用 DeepFashion 数据库开发了四个基准，包括属性预测、消费者到商店的衣服检索、店内衣服检索和地标检测。这些基准的数据和注释也可以用作以下计算机视觉任务的训练和测试集，例如衣服检测、衣服识别和图像检索。
  - 论文：[ViscoNet: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet](https://arxiv.org/pdf/2312.03154.pdf)

- **COCO（COCO Captions） Dataset**
  - 地址：https://cocodataset.org/#download
  - [简介](https://cocodataset.org/#home)：COCO Captions是一个字幕数据集，它以场景理解为目标，从日常生活场景中捕获图片数据，通过人工生成图片描述。该数据集包含330K个图文对。
  - 论文 [Text to image generation Using Generative Adversarial Networks (GANs)](https://github.com/ayansengupta17/GAN)
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/bc65a58b-175f-4088-9323-c65cd2873064)

- **CUBS-2000-2021 Dataset**
  - 地址：https://www.vision.caltech.edu/datasets/cub_200_2011/
  - 相关数据：https://www.vision.caltech.edu/datasets/
  - [简介](https://www.vision.caltech.edu/datasets/)：该数据集由加州理工学院在2010年提出的细粒度数据集，也是目前细粒度分类识别研究的基准图像数据集。该数据集共有11788张鸟类图像，包含200类鸟类子类，其中训练数据集有5994张图像，测试集有5794张图像，每张图像均提供了图像类标记信息，图像中鸟的bounding box，鸟的关键part信息，以及鸟类的属性信息。
  - 论文 [Text to image generation Using Generative Adversarial Networks (GANs)](https://github.com/ayansengupta17/GAN)
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/2fe78b0b-31f3-4373-ab94-87e57ed70de7)

- **102 Category Flower Dataset**
  - 地址：https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
  - [简介](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)：我们创建了一个 102 个类别的数据集，由 102 个花卉类别组成。这些花被选为英国常见的花。每个类别由 40 到 258 张图像组成。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/b89fa24d-2711-406f-8657-dc20587fe469)

- **Flickr8k_dataset**
  - 地址：https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
  - [简介](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)：用于基于句子的图像描述和搜索的新基准集合，由 8,000 张图像组成，每张图像都配有五个不同的标题，这些标题提供了对显着实体和事件的清晰描述。这些图像是从六个不同的 Flickr 组中选出的，往往不包含任何知名人物或地点，而是手动选择来描绘各种场景和情况
  - 论文：Caption to Image generation using Deep Residual Generative Adversarial Networks [DR-GAN]
  ```
    Flickr8k_Dataset.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
    Flickr8k_text.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
  ```

- **Nouns Dataset**自动添加标题的名词数据集卡
  - 地址：https://huggingface.co/datasets/m1guelpf/nouns
  - [简介](https://huggingface.co/datasets/m1guelpf/nouns)：用于训练名词文本到图像模型的数据集，根据名词的属性、颜色和项目自动生成名词的标题。对于每一行，数据集包含 image 和 text 键。 image 是不同大小的 PIL jpeg， text 是随附的文本标题。仅提供火车分班。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/d7222a65-d3ea-42ad-84a4-eb8a9b02e128)

- **OxfordTVG-HIC Dataset**大规模幽默图像文本数据集
  - 地址：https://github.com/runjiali-rl/Oxford_HIC?tab=readme-ov-file
  - [简介](https://github.com/runjiali-rl/Oxford_HIC?tab=readme-ov-file)：这是一个用于幽默生成和理解的大型数据集。幽默是一种抽象的、主观的、依赖于情境的认知结构，涉及多种认知因素，使其生成和解释成为一项具有挑战性的任务。Oxford HIC 提供了大约 290 万个带有幽默分数的图像文本对，以训练通用的幽默字幕模型。与现有的字幕数据集相反，Oxford HIC 具有广泛的情感和语义多样性，导致脱离上下文的示例特别有利于产生幽默。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/a304c7ee-1d90-4065-ac30-4cf1bd83d74d)

- **Multi-Modal-CelebA-HQ**大规模人脸图像文本数据集
  - 地址：https://github.com/IIGROUP/MM-CelebA-HQ-Dataset
  - [简介](https://github.com/runjiali-rl/Oxford_HIC?tab=readme-ov-file)：Multi-Modal-CelebA-HQ (MM-CelebA-HQ) 是一个大规模人脸图像数据集，其中有 30k 高分辨率人脸图像，是按照 CelebA-HQ 从 CelebA 数据集中选择的。数据集中的每个图像都附有语义掩模、草图、描述性文本和具有透明背景的图像。Multi-Modal-CelebA-HQ 可用于训练和评估一系列任务的算法，包括文本到图像生成、文本引导图像操作、草图到图像生成、图像字幕和视觉问答。该数据集被引入并在 TediGAN 中使用。
  - 样本示例 ![image](https://github.com/bolongliu/Controlled-Text-Generation-Image-Datasets/assets/92673294/3bb41c16-0209-455d-88e6-85d539636ffb)




## 3D 数据集
## 1. 预训练数据集
- Multimodal3DIdent：从可控地面真实因素生成的图像/文本对的多模态数据集
  - 地址：https://zenodo.org/records/7678231
  - 简介：ICLR 2023 上发表的《多模态对比学习的可识别性结果》一文中介绍了用于生成 Multimodal3DIdent 数据集的官方代码。该数据集提供了可识别性基准，其中包含从可控地面真实因素生成的图像/文本对，其中一些在图像和文本模态之间共享，如以下示例所示。
  - 论文：[Identifiability Results for Multimodal Contrastive Learning](https://arxiv.org/pdf/2303.09166.pdf)
## 2. 文生图微调数据集

## 3. 可控文本生成图像数据集
