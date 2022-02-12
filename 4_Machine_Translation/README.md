# Machine Translation with Transformer Networks

__Author:__ Vincent Gurgul <br>
__Course:__ Information Systems Seminar <br>
__Institute:__ Humboldt University Berlin, Chair of Information Systems <br>
__Lecturer:__ Prof. Dr. Stefan Lessmann <br>
__Semester:__ WS 2021/22 <br>

## Contents of this repository

```
.
├── Machine_Translation.ipynb     # jupyter notebook on neural machine translation
├── README.md                     # this readme file
├── requirements.txt              # requirements file for the notebook
└── test_dataset.parquet          # WMT16 test dataset of German-English sentence pairs
```

## Contents of this README

```
1. Introduction
2. Literature Review
3. Benchmarks
4. Results
5. Discussion
6. Conclusion
References
```

## 1. Introduction

The task of a translator is to express the meaning of a passage of text in a different language. Machine translation is a discipline that studies the performance of this task with the utilisation of digital computers. After decades of development, great achievements have been made in the field of machine translation in recent years with the advent of neural machine translation.

In this notebook a review of existing literature on machine translation will be conducted first. The literature review will include an overview of the history of machine translation and an introduction into the state-of-the-art neural network architecture for machine translation, the Transformer.

Then, language evaluation metrics are introduced and previous machine translation model architectures are presented, that will be used as a benchmark for the Transformer. Those include LSTM networks, CNNs and GLUs.

Finally, five different Transformer-based python libraries for machine translation will be evaluated and compared. The results of the empirical analysis and the advantages and disadvantages of each python package are presented and then possible future improvements of machine translation are discussed.

For a general introduction into the Transformer network and its architecture see our notebook 'Transformer_Introduction.ipynb', which you'll also find on our [Github](https://github.com/VincentGurgul/transformers).

## 2. Literature Review

The concept of using digital computers to translate documents between natural human languages was first mentioned by Warren Weaver in a letter to the cyberneticist Norbert Wiener in March 1947, in which he made references to cryptrography. [[Weaver, 1947]](#References)

The first publicly demonstrated machine translation project was a Russian-English translation programme for the US military, developed as a collaboration between International Business Machines Corporation (IBM) and Georgetown University between 1951 and 1954. [[IBM, 1954]](#References)

Despite its anecdotally poor quality, the programme enjoyed high popularity among US military personnel who, for the first time, could at least get an impression of the content of Russian documents themselves without the diversions via third parties. However, it also raised expectations that automatic systems will be able to produce high-quality translations in the near future, which later turned out to be quite unrealistic. [[History of Information, n.d.]](#References)

In terms of methodology, there are two approaches to machine translation. The first approach was rule-based translation, which was dominant until 1990. In rule-based translation a bilingual dictionary and manually written rules are used. However, developing those rule-based algorithms is labor intensive and the rules can't be easily transferred from one language to another, making them hard to scale for multilingual translation. [[Wang et al., 2021]](#References)

In 1990, [[Brown et al., 1990]](#References) proposed the concept of statistical machine translation, in which machines learn from a large amount of data instead of relying on human experts to write rules. With the availability of large corpora of bilingual sentence pairs, corpus-based machine translation methods became dominant since then. [[Wang et al., 2021]](#References)

In April 2006, the first internet translation service based on statistical machine translation methods was launched by Google. [[Och, 2006]](#References)

The introduction of pivot-based translation was responsible for significant improvements in the translations of low-resource languages. The idea is that the translation from a source to a target languange can be improved by introducing a third language, the pivot language, for which a larger amount of bilingual training is available. [[Wang et al., 2006]](#References)

With strong progress in deep learning technology in speech, vision, and other fields, researchers began to apply deep learning technology to machine translation. In 2014, [[Bahdanau et al., 2014]](#References) proposed an RNN-based attentional encoder-decoder neural network architecture for machine translation. The encoder maps the source sentence to a real-valued vector or matrix, from which the decoder creates the translation in the target language. In the same paper the authors also introduced the concept of the attention mechanism for the first time and coined the term “neural machine translation”. 

However, this network still had a significant weakness — the out-of-vocabulary problem — where the model is not capable of translating a word it has never encountered before. This problem was solved by [[Sennrich et al., 2015]](#References) with the introduction of byte-pair encoding as a subword-tokenization algorithm. This method made machine translation models capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units as is used in machine translation until this day.

Those advances in the field led to [[Dong et al., 2015]](#References) introducing the first multilingual neural machine translation model based on that attentional encoder-decoder architecture. This model achieved significantly higher translation quality over individually learned models for both high- and low-resource languages, by using a shared encoder.

In 2016, Google followed suit and transitioned their internet translation service to the neural machine translation system GNMT. [[Wu et al., 2016]](#References) One year later Baidu deployed its first large-scale NMT system which was based on an early version of MarianMT, an open-source translation model developed at the University of Edinburgh, Adam Mickiewicz University in Poznań and at Microsoft. [[Baidu, 2017]](#References)

Different languages have different morphologies and structures, which makes translation among many different languages a very difficult task. Chinese, for example, is a subject-verb-object language, while Japanese is a subject-object-verb language. Therefore, when translating from Chinese to Japanese, long-distance reordering is usually required. This is the most significant challenge for the recurrence-based neural networks that have been used until this point, where information decays rapidly during transmission in the network. [[Wang et al., 2021]](#References)

That issue is resolved with the introduction of the state-of-the-art neural machine translation architecture, the Transformer by [[Vaswani et al., 2017]](#References). The authors refrain from using the Attention mechanism as an intermediate module between the Encoder and the Decoder and instead make it the heart of the network by putting it directly in front of the neural network in every single encoder and decoder block. The Transformer architecture processes all words of the source language simultaneously rather than sequentially, allowing for high parallelization and theoretically infitely long memory. 

This work has laid the foundation for two important machine translation models: bidirectional encoder representations from transformers (BERT) by Google [[Devlin et al., 2018]](#References) and enhanced representation through knowledge integration (ERNIE) by Baidu. [[Sun et al., 2019]](#References)

In recent years the Transformer architecture has been applied to ever larger datasets and further developed by combining it with methods suchs as pivot-based translation, back-translation and multilingual finetuning.

This has led to the recent publishing of the best neural machine translation model to date by Facebook. [[Tran et al., 2021]](#References)

## 3. Benchmarks

In the following sections we will look at different model architectures that have been used for machine translation in existing literature. We will compare their BLEU scores on the English-German dataset from the Ninth Workshop on Statistical Machine Translation in 2014, to set a benchmark for what to expect from the modern transformer-based models that are being compared in section 4.

The Workshop on Statistical Machine Translation (WMT) is a series of annual workshops and conferences on statistical machine translation, going back to 2006. Each year various machine translation challenges are published beforehand and evaluated during the conference, including translation of news, chats, biomedical data, low-resource languages and automatic post-editing.

When publishing a new machine translation model or architecture it has become common among researchers to evaluate it on the WMT news dataset for comparabilities' sake and because it used to be and still is one of the highest quality open-source sets of multilingual sentence pairs.

As the benchmarks are already a couple years old, they were all evaluated on the WMT14 news dataset. We will be evaluating the Python libraries in sections 3.4 onwards on the WMT16 news dataset, because of its larger size and improved quality.

### Attentional LSTMs

The first architecture that will be introduced as a benchmark is a combination of LSTM networks and attention. A network of bi- or uni-directional LSTMs is used as an Encoder, a uni-directional LSTM network is used as a decoder and in between there is an attention module. This architecture has been used for machine translation by [[Luong et al., 2015]](#References) and [[Wu et al., 2016]](#References).

[[Luong et al., 2015]](#References) presented a BLEU score of 20.9 with a single model and 23.0 with an ensemble of 8 models on the WMT14 En-De dataset. The architecture they used consisted of 15 uni-directional encoder layers and 15 uni-directional decoder layers.

Google's Neural Machine Translation system (GNMT) presented by [[Wu et al., 2016]](#References) on the other hand achieved a BLEU score of 24.61 with a single model on the same dataset. GNMT uses 8 encoder layers (1 bi-directional layer and 7 uni-directional layers) and 8 uni-directional decoder layers.

### Attentional CNNs

The attentional CNN combines convolutions with gated linear units and the attention mechanism. This architecture has been proposed by [[Gehring et al., 2017]](#References) and successfully applied for machine translation. Because the architecture is based entirely on convolutional neural networks, computations can be fully parallelized during training.

This network outperformes the attentional LSTM with a BLEU of 24.61 for a single model and 26.43 for an ensemble of 8 models on the WMT14 En-De dataset.

### Transformers

Finally we look at the network architecture introduced by [[Vaswani et al., 2017]](#References), commonly referred to as the transformer. It dispenses with recurrence or convolutions entirely and makes the attention mechanism the heart of the network, only accompanied by positionwise feed-forward networks, hence the name of the paper — “Attention is all you need”.

The authors refrain from using the Attention mechanism as an intermediate module between the Encoder and the Decoder though and instead put it directly in front of the feed-forward neural network in every single encoder and decoder block. The Transformer architecture processes all words of the source language simultaneously rather than sequentially, allowing for high parallelization and allowing it to theoretically capture infinitely long relationships. 

The proposed network consist of 6 encoder layers, 6 decoder layers and 8 attention heads in each attention block. Because the network processes all inputs simultaneously, it requires positional encoding to be added to the word embeddings.

The authors also test their network on the WMT14 En-De dataset and are able to outperform all previous architectures with a BLEU score of 28.40 for a single model. This result should set the expectation for the performance of the libraries that are being compared in the following sections, as their architectures are identical or very similar to that proposed by [[Vaswani et al., 2017]](#References).

### Overview

|Reference|Architecture|BLEU WMT14 En-De|
|:-|:-|:-|
|[[Luong et al., 2015]](#References)|Attentional LSTM|0.209|
|[[Luong et al., 2015]](#References)|Attentional LSTM Ensemble|0.230|
|[[Wu et al., 2016]](#References)|Attentional LSTM|0.246|
|[[Gehring et al., 2017]](#References)|Attentional CNN|0.252|
|[[Gehring et al., 2017]](#References)|Attentional CNN Ensemble|0.264|
|[[Vaswani et al., 2017]](#References)|Transformer|0.284|

<br>

As you can see summarized in the table above, a significant improvement has been achieved when transitioning from the recurrent LSTM networks to convolution-based network architectures.

However, the greatest step has been made with the introduction of the Transformer. Note that the BLEU score of 28.4 that Vaswani et al. were able to achieve was still limited by the computational resources as well as the amount and quality of training data and sampling methods.

The most recent Transformer model which won the WMT21 news translation challenge in several languages achieved a new record of BLEU 40.8 when translating from English to German. This was achieved using multilingual training data and multilingual fine-tuning.

When using an emsemble of models and combining them with reranking and postprocessing, the authors were able to achieve a BLEU score of 42.6, beating the WMT20 winning model by 3.8 points. [[Tran et al., 2021]](#References)

Now, how do these scores compare to what a human translator would achieve? Even though the BLEU score ranges from 0 to 1, a professional human translation will rarely achieve a score of 1, unless it is identical to the reference translation(s). And this is unlikely, as professional translators don't necessarily always agree on what a 'correct' translation is.

The authors of the BLEU metric give an answer to this question in their paper [[Papineni et al., 2001]](#References):

``` 
“[...] on a test corpus of about 500 sentences (40 general news stories), a human translator scored 0.3468
against four references and scored 0.2571 against two references.”
```

Therefore, as the WMT news translation datasets only feature one reference translation, a human translator would be expected to achieve a BLEU score of less than 0.2571. This would be less than even some convolution-based networks were able to achieve in 2017. Unfortunately, the paper does not specify the skill level of the translator.


## 4. Results

|Model|Reference|Languages|Size|Sentences <br>per minute|BLEU|GLEU|hLepor|F-measure|
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
|OpusMT (pytorch)|[[Tiedemann & Thottingal, 2020]](#Tiedemann)|186|300 MB|42|0.29|0.38|0.76|0.63|
|OpusMT (C++)|[[Tiedemann & Thottingal, 2020]](#Tiedemann)|186|300 MB|100|0.29|0.38|0.76|0.63|
|mBART50|[[Tang et al., 2020]](#Tang)|52|2.3 GB|20|0.27|0.36|0.74|0.62|
|M2M-100-1.2B|[[Fan et al., 2020]](#Fan)|100|5.0 GB|27|0.28|0.38|0.76|0.63|
|WMT19 Winner|[[Ng et al., 2019]](#Ng)|3|11.9 GB|30|0.37|0.45|0.79|0.68|

<br>

The WMT19 Winner clearly outperformed the other models, however the superiority in terms of hLepor is lower than in terms of the other measures. It’s hard to recommend it though, because it’s only translating between German, English and Russian. The winners of the WMT19 translation challenges for other languages have not open-sourced their models.

Therefore, the best recommendation for multilingual translation is the OpusMT model in the C++ implementation. It’s by far the fastest, covers the most languages and still has a slight advantage over the other multilingual models in terms of performance.

## 5. Discussion

Equality of people is based, among other things, on the equal opportunity to access information. This becomes a particularly important goal at a time when digital information channels are becoming the most important way of integrating into modern society. In this context, language barriers can lead to disadvantages and misunderstandings. Machine translation has evolved into an important tool to overcome these language barriers.

Some metrics may even give the impression that computers are better at translation than humans. However, these metrics do not necessarily capture the whole picture. At this point in time, computers can generate very good and fluent translations for some languages. However, the models reach their limits in the area of simultaneous translation or belletrisitic, for example.

In simultaneous translation, the human translator does not have the task of translating everything exactly. He knows what he should focus on and what he can leave out. Machine translation systems, on the other hand, translate everything and are not yet able to skip irrelvant parts in order to reduce the translation time. Furthermore, a human translator is able to account for body language or if a speaker is referencing slides that he presents.

Some translators, when translating Proust's seven-volume work “À la Recherche du Temps Perdu”, have sought to make the first word of the first volume the same as the last word of the last volume because the French original begins and ends with the same word. [[Craig, 2020]](#References)

Translation, then, in its most advanced form, involves close study of the original text and sometimes even the author's life story and circumstances. These are facets that neural machine translation systems are not yet able to conceive.

Furthermore, machine translation systems require a significantly larger amount of data to learn a language than humans. Although some methods have been proposed to improve the learning of low-resource languages, they remain a significant challenge for machine translation systems.

The robustness of machine translation systems also leaves a fair amount to be desired. Even a wrong punctuation mark can have a considerable influence on the resulting translation. Humans are still much better at dealing with errors in the source text and correcting them quickly.

In summary, there is still room for improvement in machine translation and it can be assumed that many innovations await us in the coming years.

## 6. Conclusion

The main purpose of this notebook was to present and evaluate the current state of the art of neural machine translation with Transformer-based libraries in Python. In the beginning, an overview of the history of machine translation has been presented in the form of a literature review. The current most advanced neural machine translation architecture, the Transformer, has been introduced.

In addition to relevant metrics for language evaluation, the architectures of neural networks, that were used for machine transalation before the Transformer, were presented. Those included attentional LSTM networks and attentional convolutional networks. The BLEU scores that the authors of these neural network architectures have been able to achieve were introduced as a benchmark for the performance of the Transformer-based Python libraries.

In the main part, five Transformer-based Python libraries have been presented. Both single sentence and for corpus-level translations have been demonstrated for each model. In this conjunction, language evaluation scores were computed on a dataset of 3000 German-English sentence pairs from the WMT news data translation challenge in 2016, thus displaying the performance of the models on real-world data.

While Facebook has repeatedly introduced models in recent years that have raised the bar for scores in the common language evaluation metrics for certain languages in the scope of the WMT conference, they have probably done so primarily in order to profile themselves. These high-performance models are only useful for translation in a handful of languages and therefore cannot be considered the spearhead of neural machine translation in general.

When it comes to many-to-many multilingual translation, Facebook's M2M-100 model has demonstrated good performance. However, it was shown, that the C++ based implementation of OpusMT remains the best open-source option for multilingual machine translation in Python.

Overall, the results demonstrate that neural network based machine translation systems are able to produce translations that can largely compete with human translations within a much shorter span of time than a human would require.

# References

<a id='Bahdanau'>[Bahdanau et al., 2014]</a>

&ensp;&ensp;&ensp; _Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. ArXiv.org. https://arxiv.org/abs/1409.0473_

<a id='Baidu'>[Baidu, 2017]</a>

&ensp;&ensp;&ensp; _Baidu. (2017). WIPO Translate: Terms and Conditions for the Usage and User Guide A. What is it? https://patentscope.wipo.int/translate/wtapta-user-manual-en.pdf_

<a id='Brown'>[Brown et al., 1990]</a>

&ensp;&ensp;&ensp; _Brown, P. F., Cocke, J., Della Pietra, S. A., Della Pietra, V. J., Jelinek, F., Lafferty, J. D., Mercer, R. L., & Roossin, P. S. (1990). A Statistical Approach to Machine Translation. Computational Linguistics, 16(2), 79–85. https://aclanthology.org/J90-2002/_

<a id='Craig'>[Craig, 2020]</a>

&ensp;&ensp;&ensp; _Craig, H. E. (2020). Assessing the English and Spanish translations of Proust’s A la recherche du temps perdu. Peter Lang Publishing, Inc._

<a id='Devlin'>[Devlin et al., 2018]</a>

&ensp;&ensp;&ensp; _Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. ArXiv.org. https://arxiv.org/abs/1810.04805_

<a id='Dong'>[Dong et al., 2015]</a>

&ensp;&ensp;&ensp; _Dong, D., Wu, H., He, W., Yu, D., & Wang, H. (2015, July 1). Multi-Task Learning for Multiple Language Translation. ACLWeb; Association for Computational Linguistics. https://doi.org/10.3115/v1/P15-1166_

<a id='Fan'>[Fan et al., 2020]</a>

&ensp;&ensp;&ensp; _Fan, A., Bhosale, S., Schwenk, H., Ma, Z., El-Kishky, A., Goyal, S., Baines, M., Celebi, O., Wenzek, G., Chaudhary, V., Goyal, N., Birch, T., Liptchinsky, V., Edunov, S., Grave, E., Auli, M., & Joulin, A. (2020). Beyond English-Centric Multilingual Machine Translation. ArXiv:2010.11125 [Cs]. https://arxiv.org/abs/2010.11125_

<a id='Gehring'>[Gehring et al., 2017]</a>

&ensp;&ensp;&ensp; _Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Yann N. (2017). Convolutional Sequence to Sequence Learning. ArXiv.org. https://arxiv.org/abs/1705.03122_

<a id='Han'>[Han et al., 2013]</a>

&ensp;&ensp;&ensp; _Han, F., Wong, D., Mo, D., Chao, L., Lu, Y., Wang, Y., & Zhou, J. (2013). A Description of Tunable Machine Translation Evaluation Systems in WMT13 Metrics Task (pp. 414–421). https://aclanthology.org/W13-2253.pdf_

<a id='History'>[History of Information, n.d.]</a>

&ensp;&ensp;&ensp; _History of Information. (n.d.). The First Public Demonstration of Machine Translation Occurs : History of Information. Www.historyofinformation.com. Retrieved February 8, 2022, from https://www.historyofinformation.com/detail.php?id=666_

<a id='IBM'>[IBM, 1954]</a>

&ensp;&ensp;&ensp; _IBM. (1954, January 8). IBM Archives: 701 Translator. Www.ibm.com. https://www.ibm.com/ibm/history/exhibits/701/701_translator.html_

<a id='Junczys'>[Junczys-Dowmunt et al., 2018]</a>

&ensp;&ensp;&ensp; _Junczys-Dowmunt, M., Grundkiewicz, R., Dwojak, T., Hoang, H., Heafield, K., Neckermann, T., Seide, F., Germann, U., Aji, A. F., Bogoychev, N., Martins, A. F. T., & Birch, A. (2018, July 1). Marian: Fast Neural Machine Translation in C++. ACLWeb; Association for Computational Linguistics. https://doi.org/10.18653/v1/P18-4020_

<a id='Luong'>[Luong et al., 2015]</a>

&ensp;&ensp;&ensp; _Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. ArXiv.org. https://arxiv.org/abs/1508.04025_

<a id='Melamed'>[Melamed et al., 2003]</a>

&ensp;&ensp;&ensp; _Melamed, I. D., Green, R., & Turian, J. P. (2003). Precision and Recall of Machine Translation. ACLWeb. https://aclanthology.org/N03-2021/_

<a id='MUC'>[MUC-4, 1992]</a>

&ensp;&ensp;&ensp; _FOURTH MESSAGE UNDERSTANDING CONFERENCE (MUC-4). (1992). https://www.aclweb.org/anthology/M92-1000.pdf_

<a id='Mutton'>[Mutton et al., 2007]</a>

&ensp;&ensp;&ensp; _Mutton, A., Dras, M., Wan, S., & Dale, R. (2007, June 1). GLEU: Automatic Evaluation of Sentence-Level Fluency. ACLWeb; Association for Computational Linguistics. https://aclanthology.org/P07-1044/_

<a id='Ng'>[Ng et al., 2019]</a>

&ensp;&ensp;&ensp; _Ng, N., Yee, K., Baevski, A., Ott, M., Auli, M., & Edunov, S. (2019). Facebook FAIR’s WMT19 News Translation Task Submission. ArXiv:1907.06616 [Cs]. https://arxiv.org/abs/1907.06616_

<a id='Och'>[Och, 2006]</a>

&ensp;&ensp;&ensp; _Och, F. (2006, April 28). Statistical machine translation live. Google AI Blog. https://ai.googleblog.com/2006/04/statistical-machine-translation-live.html_

<a id='Papineni'>[Papineni et al., 2001]</a>

&ensp;&ensp;&ensp; _Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2001). BLEU. Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02. https://doi.org/10.3115/1073083.1073135_

<a id='Sennrich'>[Sennrich et al., 2015]</a>

&ensp;&ensp;&ensp; _Sennrich, R., Haddow, B., & Birch, A. (2015). Neural Machine Translation of Rare Words with Subword Units. ArXiv.org. https://arxiv.org/abs/1508.07909_

<a id='Sun'>[Sun et al., 2019]</a>

&ensp;&ensp;&ensp; _Sun, Y., Wang, S., Li, Y., Feng, S., Tian, H., Wu, H., & Wang, H. (2019). ERNIE 2.0: A Continual Pre-training Framework for Language Understanding. ArXiv.org. https://arxiv.org/abs/1907.12412_

<a id='Tang'>[Tang et al., 2020]</a>

&ensp;&ensp;&ensp; _Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J., & Fan, A. (2020). Multilingual Translation with Extensible Multilingual Pretraining and Finetuning. ArXiv:2008.00401 [Cs]. https://arxiv.org/abs/2008.00401_

<a id='Tiedemann'>[Tiedemann & Thottingal, 2020]</a>

&ensp;&ensp;&ensp; _Tiedemann, J., & Thottingal, S. (2020, November 1). OPUS-MT – Building open translation services for the World. ACLWeb; European Association for Machine Translation. https://aclanthology.org/2020.eamt-1.61/_

<a id='Tran'>[Tran et al., 2021]</a>

&ensp;&ensp;&ensp; _Tran, C., Bhosale, S., Cross, J., Koehn, P., Edunov, S., & Fan, A. (2021). Facebook AI WMT21 News Translation Task Submission. ArXiv:2108.03265 [Cs]. https://arxiv.org/abs/2108.03265_

<a id='Vaswani'>[Vaswani et al., 2017]</a>

&ensp;&ensp;&ensp; _Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, Aidan N, Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. ArXiv.org. https://arxiv.org/abs/1706.03762_

<a id='Wang2021'>[Wang et al., 2021]</a>

&ensp;&ensp;&ensp; _Wang, H., Wu, H., He, Z., Huang, L., & Church, K. W. (2021). Progress in Machine Translation. Engineering. https://doi.org/10.1016/j.eng.2021.03.023_

<a id='Wang2006'>[Wang et al., 2006]</a>

&ensp;&ensp;&ensp; _Wang, H., Wu, H., & Liu, Z. (2006, July 1). Word Alignment for Languages with Scarce Resources Using Bilingual Corpora of Other Language Pairs. ACLWeb; Association for Computational Linguistics. https://aclanthology.org/P06-2112/_

<a id='Weaver'>[Weaver, 1947]</a>

&ensp;&ensp;&ensp; _Weaver, W. (1947, March). Warren Weaver Suggests Applying Cryptanalysis Techniques to Translation : History of Information. Www.historyofinformation.com. https://www.historyofinformation.com/detail.php?id=2990_

<a id='Wikipedia'>[Wikipedia, 2019]</a>

&ensp;&ensp;&ensp; _Wikipedia. (2019, April 19). Precision and Recall. Wikipedia; Wikimedia Foundation. https://en.wikipedia.org/wiki/Precision_and_recall_

<a id='WMT'>[WMT, 2014]</a>

&ensp;&ensp;&ensp; _WMT. (2014, June). Www.statmt.org. https://www.statmt.org/wmt14/
ACL 2014 Ninth Workshop on Statistical Machine Translation_

<a id='Wolf'>[Wolf et al., 2020]</a>

&ensp;&ensp;&ensp; _Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020, October 1). Transformers: State-of-the-Art Natural Language Processing. GitHub. https://github.com/huggingface/transformers/blob/master/src/transformers/models/marian/convert_marian_to_pytorch.py_

<a id='Wu'>[Wu et al., 2016]</a>

&ensp;&ensp;&ensp; _Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., Klingner, J., Shah, A., Johnson, M., Liu, X., Kaiser, Ł., Gouws, S., Kato, Y., Kudo, T., Kazawa, H., & Stevens, K. (2016). Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. ArXiv.org. https://arxiv.org/abs/1609.08144_
