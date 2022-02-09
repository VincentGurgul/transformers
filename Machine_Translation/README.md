# Machine Translation with Transformer Networks

__Author:__ Vincent Gurgul <br>
__Course:__ Information Systems Seminar <br>
__Institute:__ Humboldt University Berlin, Chair of Information Systems <br>
__Lecturer:__ Prof. Dr. Stefan Lessmann <br>
__Semester:__ WS 2021/22 <br>

## Contents of this repository

```
.
├── Machine_Translation.ipynb     # notebook text generation
├── requirements.txt              # requirements file for the notebook
├── test_dataset.parquet          # WMT16 test dataset of German-English sentence pairs
└── README.md                     # this readme file

```

## Contents of the README

```
1. Introduction
2. Literature Review
3. Results of the Empirical Analysis
4. Discussion
5. Conclusion
References
```

## 1. Introduction

After decades of development, great achievements have been made in the field of machine translation in recent years with the advent of neural machine translation.

In this notebook a review of existing literature on machine translation will be conducted first. The literature review will include an overview of the history of machine translation and an introduction into the state-of-the-art neural network architecture for machine translation, the Transformer.

Then, language evaluation metrics are introduced and previous machine translation model architectures are presented, that will be used as a benchmark for the Transformer. Those include LSTM networks, CNNs and GRUs.

Finally, five different Transformer-based python libraries for machine translation will be evaluated and compared. The results of the empirical analysis and the advantages and disadvantages of each python package are presented and then possible future improvements of machine translation are discussed.

## 2. Literature Review

The concept of using digital computers to translate documents between natural human languages was first mentioned by Warren Weaver in a letter to the cyberneticist Norbert Wiener in March 1947, in which he made references to cryptrography. [[Weaver, 1947]](#References)

The first publicly demonstrated machine translation project was a Russian-English translation programme for the US military, developed as a collaboration between International Business Machines Corporation (IBM) and Georgetown University between 1951 and 1954. [[IBM, 1954]](#References)

Despite its anecdotally poor quality, the programme enjoyed high popularity among US military personnel who, for the first time, could at least get an impression of the content of Russian documents themselves without the diversions via third parties. However, it also raised expectations that automatic systems will be able to produce high-quality translations in the near future, which later turned out to be quite unrealistic. [[History of Information, n.d.]](#References)

In terms of methodology, there are two approaches to machine translation. The first approach was rule-based translation, which was dominant until 1990. In rule-based translation a bilingual dictionary and manually written rules are used. However, developing those rule-based algorithms is labor intensive and the rules can't be easily transferred from one language to another, making them hard to scale for multilingual translation. [[Wang et al., 2021]](#References)

In 1990, [[Brown et al., 1990]](#References) proposed the concept of statistical machine translation, in which machines learn from a large amount of data instead of relying on human experts to write rules. With the availability of large corpora of bilingual sentence pairs, corpus-based machine translation methods became dominant since then. [[Wang et al., 2021]](#References) In April 2006, the first internet translation service based on statistical machine translation methods was launched by Google. [[Och, 2006]](#References)

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

## 3. Results of the Empirical Analysis

|Model|Sentences <br>per minute|BLEU|GLEU|hLepor|F-measure|
|:-|:-|:-|:-|:-|:-|
|OpusMT (pytorch)|42|0.29|0.38|0.76|0.63|
|OpusMT (C++)|100|0.29|0.38|0.76|0.63|
|WMT19 Winner|30|0.37|0.45|0.79|0.68|
|M2M-100-1.2B|27|0.28|0.38|0.76|0.63|
|mBART50|20|0.27|0.36|0.74|0.62|

<br>

The WMT19 Winner clearly outperformed the other models, however the superiority in terms of hLepor is lower than in terms of the other measures. It’s hard to recommend it though, because it’s only translating between German and English, and the other winners of the WMT19 translation challenge have not open-sourced their models.

So the best recommendation is the OpusMT model in the C++ implementation. It’s by far the fastest, covers the most languages and still has a slight advantage over the other multilingual models in terms of performance.

## 4. Discussion

Some metrics can give the impression that computers are better at translation than humans. However, these metrics do not necessarily capture the whole picture. At this point in time, computers can generate very good and fluent translations for some languages. However, the models reach their limits in the area of simultaneous translation, for example.

In simultaneous translation, the human translator does not have the task of translating everything exactly. He knows what he should focus on and what he can leave out. Machine translation systems, on the other hand, translate everything and are not yet able to skip irrelvant parts in order to reduce the translation time. Furthermore, a human translator is able to account for body language or if a speaker is referencing slides that he presents.

The robustness of machine translation systems also leaves much to be desired. Even a wrong punctuation mark can have a considerable influence on the resulting translation. Humans are still much better at dealing with errors in the source text and correcting them quickly.

Furthermore, machine translation systems require a significantly larger amount of data to learn a language than humans. Although some methods have been proposed to improve the learning of low-resource languages, they remain a significant challenge for machine translation systems.

In summary, there is still room for improvement in machine translation and it can be assumed that many innovations await us in the coming years.

## 5. Conclusion

The main purpose of this notebook was to present and evaluate the current state of the art of neural machine translation with Transformer-based libraries in Python. In the beginning, an overview of the history of machine translation has been presented in the form of a literature review. The current most advanced neural machine translation architecture, the Transformer, has been introduced.

In addition to relevant metrics for language evaluation, the architectures of neural networks, that were used for machine transalation before the Transformer, were presented. Those included attentional LSTM networks and attentional convolutional networks. The BLEU scores that the authors of these neural network architectures have been able to achieve were introduced as a benchmark for the performance of the Transformer-based Python libraries.

In the main part, five Transformer-based Python libraries have been presented. Both single sentence and for corpus-level translations have been demonstrated for each model. In this conjunction, language evaluation scores were computed on a dataset of 3000 German-English sentence pairs from the WMT news data translation challenge in 2016, thus displaying the performance of the models on real-world data.

While Facebook has repeatedly introduced models in recent years that have raised the bar for scores in the common language evaluation metrics for certain languages in the scope of the WMT conference, they have probably done so primarily in order to profile themselves. These high-performance models are only useful for translation in a handful of languages and therefore cannot be considered the spearhead of neural machine translation in general.

When it comes to many-to-many multilingual translation, Facebook's M2M-100 model has demonstrated good performance. However, it was shown, that the C++ based implementation of OpusMT remains the best open-source option for multilingual machine translation in Python.

Overall, the results demonstrate that neural network based machine translation systems are able to produce translations that can largely compete with human translations within a much shorter span of time than a human would require.

## References

<a id='Papieni'>[Papineni et al., 2001]</a>

&ensp;&ensp;&ensp; _Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2001). BLEU. Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02. https://doi.org/10.3115/1073083.1073135_
