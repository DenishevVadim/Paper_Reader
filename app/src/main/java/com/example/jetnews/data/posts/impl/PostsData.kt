/*
 * Copyright 2022 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@file:Suppress("ktlint:max-line-length") // String constants read better
package com.example.jetnews.data.posts.impl

import com.example.jetnews.R
import com.example.jetnews.model.Markup
import com.example.jetnews.model.MarkupType
import com.example.jetnews.model.Metadata
import com.example.jetnews.model.Paragraph
import com.example.jetnews.model.ParagraphType
import com.example.jetnews.model.Post
import com.example.jetnews.model.PostAuthor
import com.example.jetnews.model.PostsFeed
import com.example.jetnews.model.Publication

/**
 * Define hardcoded posts to avoid handling any non-ui operations.
 */

val pietro = PostAuthor("Nicolas Carion", "https://medium.com/@pmaggi")
val manuel = PostAuthor("Alec Radford", "https://medium.com/@manuelvicnt")
val florina = PostAuthor(
    "Huiyu Wang1",
    "https://medium.com/@florina.muntenescu"
)

val publication = Publication(
    "Deep Learning Papers",
    "https://github.com/facebookresearch/detr/blob/main/.github/DETR.png"
)
val paragraphsPost1 = listOf(
    Paragraph(
        ParagraphType.Header,
        "Abstract."
    ),
    Paragraph(
        ParagraphType.Text,
        "We present a new method that views object detection as a\n" +
                "direct set prediction problem. Our approach streamlines the detection\n" +
                "pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation\n" +
                "that explicitly encode our prior knowledge about the task. The main\n" +
                "ingredients of the new framework, called DEtection TRansformer or\n" +
                "DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given\n" +
                "a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output\n" +
                "the final set of predictions in parallel. The new model is conceptually\n" +
                "simple and does not require a specialized library, unlike many other\n" +
                "modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation\n" +
                "in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at\n" +
                "https://github.com/facebookresearch/detr."
    ),
    Paragraph(
        ParagraphType.Header,
        "Conclusion."
    ),
    Paragraph(
        ParagraphType.Text,
        "We presented DETR, a new design for object detection systems based on transformers and bipartite matching loss for direct set prediction. The approach\n" +
                "achieves comparable results to an optimized Faster R-CNN baseline on the challenging COCO dataset. DETR is straightforward to implement and has a flexible\n" +
                "architecture that is easily extensible to panoptic segmentation, with competitive\n" +
                "results. In addition, it achieves significantly better performance on large objects\n" +
                "than Faster R-CNN, likely thanks to the processing of global information performed by the self-attention.\n" +
                "This new design for detectors also comes with new challenges, in particular\n" +
                "regarding training, optimization and performances on small objects. Current\n" +
                "detectors required several years of improvements to cope with similar issues,\n" +
                "and we expect future work to successfully address them for DETR."
    ),
    Paragraph(
        ParagraphType.Header,
        "PyTorch inference code."
    ),
    Paragraph(
        ParagraphType.Text,
        "To demonstrate the simplicity of the approach, we include inference code with\n" +
                "PyTorch and Torchvision libraries in Listing 1. The code runs with Python 3.6+,\n" +
                "PyTorch 1.4 and Torchvision 0.5. Note that it does not support batching, hence\n" +
                "it is suitable only for inference or training with DistributedDataParallel with\n" +
                "one image per GPU. Also note that for clarity, this code uses learnt positional\n" +
                "encodings in the encoder instead of fixed, and positional encodings are added\n" +
                "to the input only instead of at each transformer layer. Making these changes\n" +
                "requires going beyond PyTorch implementation of transformers, which hampers\n" +
                "readability. The entire code to reproduce the experiments will be made available\n" +
                "before the conference."
    ),
    Paragraph(
        ParagraphType.CodeBlock,
        "1 import torch\n" +
                "2 from torch import nn\n" +
                "3 from torchvision.models import resnet50\n" +
                "4\n" +
                "5 class DETR(nn.Module):\n" +
                "6\n" +
                "7 def __init__(self, num_classes, hidden_dim, nheads,\n" +
                "8 num_encoder_layers, num_decoder_layers):\n" +
                "9 super().__init__()\n" +
                "10 # We take only convolutional layers from ResNet-50 model\n" +
                "11 self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])\n" +
                "12 self.conv = nn.Conv2d(2048, hidden_dim, 1)\n" +
                "13 self.transformer = nn.Transformer(hidden_dim, nheads,\n" +
                "14 num_encoder_layers, num_decoder_layers)\n" +
                "15 self.linear_class = nn.Linear(hidden_dim, num_classes + 1)\n" +
                "16 self.linear_bbox = nn.Linear(hidden_dim, 4)\n" +
                "17 self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))\n" +
                "18 self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))\n" +
                "19 self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))\n" +
                "20\n" +
                "21 def forward(self, inputs):\n" +
                "22 x = self.backbone(inputs)\n" +
                "23 h = self.conv(x)\n" +
                "24 H, W = h.shape[-2:]\n" +
                "25 pos = torch.cat([\n" +
                "26 self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),\n" +
                "27 self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),\n" +
                "28 ], dim=-1).flatten(0, 1).unsqueeze(1)\n" +
                "29 h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),\n" +
                "30 self.query_pos.unsqueeze(1))\n" +
                "31 return self.linear_class(h), self.linear_bbox(h).sigmoid()\n" +
                "32\n" +
                "33 detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)\n" +
                "34 detr.eval()\n" +
                "35 inputs = torch.randn(1, 3, 800, 1200)\n" +
                "36 logits, bboxes = detr(inputs)"
    ),
)

val paragraphsPost2 = listOf(
    Paragraph(
        ParagraphType.Header,
        "Abstract."
    ),
    Paragraph(
        ParagraphType.Text,
        "State-of-the-art computer vision systems are\n" +
                "trained to predict a fixed set of predetermined\n" +
                "object categories. This restricted form of supervision limits their generality and usability since\n" +
                "additional labeled data is needed to specify any\n" +
                "other visual concept. Learning directly from raw\n" +
                "text about images is a promising alternative which\n" +
                "leverages a much broader source of supervision.\n" +
                "We demonstrate that the simple pre-training task\n" +
                "of predicting which caption goes with which image is an efficient and scalable way to learn SOTA\n" +
                "image representations from scratch on a dataset\n" +
                "of 400 million (image, text) pairs collected from\n" +
                "the internet. After pre-training, natural language\n" +
                "is used to reference learned visual concepts (or\n" +
                "describe new ones) enabling zero-shot transfer\n" +
                "of the model to downstream tasks. We study\n" +
                "the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and\n" +
                "many types of fine-grained object classification.\n" +
                "The model transfers non-trivially to most tasks\n" +
                "and is often competitive with a fully supervised\n" +
                "baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet\n" +
                "zero-shot without needing to use any of the 1.28\n" +
                "million training examples it was trained on. We\n" +
                "release our code and pre-trained model weights at\n" +
                "https://github.com/OpenAI/CLIP."
    ),
    Paragraph(
        ParagraphType.Header,
        "Conclusion."
    ),
    Paragraph(
        ParagraphType.Text,
        "We have investigated whether it is possible to transfer the\n" +
                "success of task-agnostic web-scale pre-training in NLP to\n" +
                "another domain. We find that adopting this formula results in similar behaviors emerging in the field of computer\n" +
                "vision and discuss the social implications of this line of\n" +
                "research. In order to optimize their training objective, CLIP\n" +
                "models learn to perform a wide variety of tasks during pretraining. This task learning can then be leveraged via natural\n" +
                "language prompting to enable zero-shot transfer to many\n" +
                "existing datasets. At sufficient scale, the performance of this\n" +
                "approach can be competitive with task-specific supervised\n" +
                "models although there is still room for much improvement.\n" +
                "ACKNOWLEDGMENTS\n" +
                "We’d like to thank the millions of people involved in creating\n" +
                "the data CLIP is trained on. We’d also like to thank Susan\n" +
                "Zhang for her work on image conditional language models\n" +
                "while at OpenAI, Ishaan Gulrajani for catching an error in\n" +
                "the pseudocode, and Irene Solaiman, Miles Brundage, and\n" +
                "Gillian Hadfield for their thoughtful feedback on the broader\n" +
                "impacts section of the paper. We are also grateful to the\n" +
                "Acceleration and Supercomputing teams at OpenAI for their\n" +
                "critical work on software and hardware infrastructure this\n" +
                "project used. Finally, we’d also like to thank the developers\n" +
                "of the many software packages used throughout this project\n" +
                "including, but not limited, to Numpy (Harris et al., 2020),\n" +
                "SciPy (Virtanen et al., 2020), ftfy (Speer, 2019), TensorFlow (Abadi et al., 2016), PyTorch (Paszke et al., 2019),\n" +
                "pandas (pandas development team, 2020), and scikit-learn\n" +
                "(Pedregosa et al., 2011)."
    ),

)

val paragraphsPost3 = listOf(
    Paragraph(
        ParagraphType.Header,
        "Abstract."
    ),
    Paragraph(
        ParagraphType.Text,
        "We present MaX-DeepLab, the first end-to-end model for\n" +
                "panoptic segmentation. Our approach simplifies the current pipeline that depends heavily on surrogate sub-tasks\n" +
                "and hand-designed components, such as box detection, nonmaximum suppression, thing-stuff merging, etc. Although\n" +
                "these sub-tasks are tackled by area experts, they fail to\n" +
                "comprehensively solve the target task. By contrast, our\n" +
                "MaX-DeepLab directly predicts class-labeled masks with a\n" +
                "mask transformer, and is trained with a panoptic quality inspired loss via bipartite matching. Our mask transformer\n" +
                "employs a dual-path architecture that introduces a global\n" +
                "memory path in addition to a CNN path, allowing direct\n" +
                "communication with any CNN layers. As a result, MaXDeepLab shows a significant 7.1% PQ gain in the box-free\n" +
                "regime on the challenging COCO dataset, closing the gap\n" +
                "between box-based and box-free methods for the first time.\n" +
                "A small variant of MaX-DeepLab improves 3.0% PQ over\n" +
                "DETR with similar parameters and M-Adds. Furthermore,\n" +
                "MaX-DeepLab, without test time augmentation, achieves\n" +
                "new state-of-the-art 51.3% PQ on COCO test-dev set."
    ),
    Paragraph(
        ParagraphType.Header,
        "Conclusion."
    ),
    Paragraph(
        ParagraphType.Text,
        "In this work, we have shown for the first time that panoptic segmentation can be trained end-to-end. Our MaXDeepLab directly predicts masks and classes with a mask transformer, removing the needs for many hand-designed\n" +
                "priors such as object bounding boxes, thing-stuff merging,\n" +
                "etc. Equipped with a PQ-style loss and a dual-path transformer, MaX-DeepLab achieves the state-of-the-art result\n" +
                "on the challenging COCO dataset, closing the gap between\n" +
                "box-based and box-free methods for the first time."
    ),
)

val post1 = Post(
    id = "dc523f0ed25c",
    title = "End-to-End Object Detection with Transformers",
    subtitle = "Nicolas Carion, \n" +
            "Francisco Massa, \n" +
            "Gabriel Synnaeve, Nicolas Usunier,\n" +
            "Alexander Kirillov, and Sergey Zagoruyko",
    url = "https://arxiv.org/pdf/2005.12872.pdf",
    publication = publication,
    metadata = Metadata(
        author = pietro,
        date = "28 May 2020",
        readTimeMinutes = 1
    ),
    paragraphs = paragraphsPost1,
    imageId = R.drawable.__2023_01_14__11_44_26,
    imageThumbId = R.drawable.__2023_01_14__11_44_26
)

val post2 = Post(
    id = "7446d8dfd7dc",
    title = "Learning Transferable Visual Models From Natural Language Supervision",
    subtitle = "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, \n" +
            "Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, \n" +
            "Ilya Sutskever",
    url = "https://arxiv.org/pdf/2103.00020.pdf",
    publication = publication,
    metadata = Metadata(
        author = manuel,
        date = "26 Feb 2021",
        readTimeMinutes = 3
    ),
    paragraphs = paragraphsPost2,
    imageId = R.drawable.__2023_01_14__11_46_55,
    imageThumbId = R.drawable.__2023_01_14__11_46_55
)

val post3 = Post(
    id = "ac552dcc1741",
    title = "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers",
    subtitle = "Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen,\n" +
            "Johns Hopkins University, Google Research",
    url = "https://arxiv.org/pdf/2012.00759.pdf",
    publication = publication,
    metadata = Metadata(
        author = florina,
        date = "12 Jul 2021",
        readTimeMinutes = 1
    ),
    paragraphs = paragraphsPost3,
    imageId = R.drawable.__2023_01_14__11_58_14,
    imageThumbId = R.drawable.__2023_01_14__11_58_14
)

val posts: PostsFeed =
    PostsFeed(
        highlightedPost = post1,
        recommendedPosts = listOf(post1, post2, post3),
        popularPosts = listOf(
        ),
        recentPosts = listOf(
        )
    )
