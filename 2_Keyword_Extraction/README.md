# README for Keyword Extraction

# Keyword Extraction with Transformer Networks

__Author:__ Iliyana Tarpova <br>
__Course:__ Information Systems Seminar <br>
__Institute:__ Humboldt University Berlin, Chair of Information Systems <br>
__Lecturer:__ Prof. Dr. Stefan Lessmann <br>
__Semester:__ WS 2021/22 <br>
__Submission Date:__ February 13, 2022 <br>

## Contents of this repository

```
.
├── Keyword_Extraction.ipynb            # jupyter notebook on keyword extraction
├── README.md                           # this readme file
├── kp20k_testing.json                  # dataset used for the evaluation
└── keyword_extraction_wordwise.py      # code for running the keyword extraction on CPU with Wordwise
```

## Contents of this README

```
1. Introduction
2. Literature Review
3. Experiments
4. Empirical Results
5. Discussion
6. Conclusion
References
```

## 1. Introduction

In this part, the use cases Text Classification and more precisely the downstream task Keyword Extraction will be presented. Text Classification is one of the important and typical tasks in supervised machine learning. It can be described as the process of categorizing text into organized groups. By using Natural Language Processing (NLP), text classifiers can automatically analyse text and then assign a set of pre-defined tags or categories based on its content. There are many interesting applications for text classification such as Question Answering, Sentiment Analysis, Text Summarization, Spam Detection, Topic Detection, and others. In this notebook, the downstream task Keyword Extraction with Transformer models will be presented. Keyword Extraction is the task of selecting words or phrases that best represent a text. It can also be described as an attempt to summarize the meaning of a document within just a few words. Formally speaking, this is an NLP task, whose aim is to embed a given paragraph once as a whole and once n-grams separately and then calculate the distances between the candidate embeddings and the text. The embeddings of the top k words that are closest to this of the text are chosen to be the keywords.

**The Process of Keyword Extraction**

The first step to extract keywords is to set the size of the n-grams. Usually, it is between one and three words, very rarely four. Then a list of all possible n-grams from the text is created. It is important that all verbs, prepositions, and adjectives are removed, as they are almost never given as key phrases by authors. This leaves us with a list of n-grams of nouns, called candidates. The candidates and the text are embedded using a language model. Here, many different language models can be used. They vary in their architecture and performance. Below, the most widely used models for keyword extraction and their differences, together with their strengths and weaknesses will be discussed. Choosing the right model is crucial and not trivial. How well they perform usually depends on the specific task and use case. After the embeddings are computed, the semantic distance between word and text embeddings is measured, normally using cosine similarity. Lastly, those n-grams are chosen, whose distance is closest to the text embedding. Similar techniques exist to improve the chosen key phrases. It is often the case, that synonyms are chosen as keywords since their distances to the text embeddings are similar. However, synonyms are not vey useful as key phrases. Therefore, diversity of the output should be ensured. One way to do that is by using the Maximal Margin Relevance (MMR) criterion [Carbonell, 1998]. It achieves that by selecting the keywords that are the most similar to the document and for each next candidate its similarity to the already chosen keywords is considered as well. The other technique is called Max Sum Similarity (MSS) and it strives to reduce redundancy by extracting the combinations that are the least similar to each other by maximizing the similarity to the text while minimizing the similarity between the candidates. Unfortunately, both strategies come with a trade-off between diversity and accuracy.

In the following section the background of automated keyword extraction will be presented together with other ways to approach this task using basic statistical methods, transformer-based models or both combined. The four different language models will be presented. With the help of the python package WordWise, keyword extraction will be showcased and the ease of use of this package will be presented. Based on this package and the data set KP20K the models will be evaluated and compared. Afterward, the results will be discussed and finally the findings of this work will be concluded.

## 2. Literature Review

Automated Keyword Extraction has been of interest to developers for quite some time. Hence, numerous approaches exist to find phrases that provide a compact representation of a document’s content in an automated manner. Previous work on document-oriented methods of keyword extraction has combined natural language processing approaches to identify part-of-speech (POS) tags that are combined with supervised learning, machine-learning algorithms, or statistical methods. [Rose, 2010]

One of the base algorithms for keyword extraction is using TF-IDF, short for term frequency–inverse document frequency. The term frequency is calculated by simply counting the occurrences of a word in a document. There exist multiple ways to adjust the TF, for example by capturing the raw count of the words in comparison to the length of the text. The inverse document frequency calculates how common a word is in a text corpus, such as Wikipedia, by dividing the number of the documents that contain a word by the total number of documents. The result is that the logarithm of the division. The TF-IDF is conducted by multiplying TF and IDF and the higher the score, the more relevant the word is for that document.

Matsuo and Ishizuka (2004) apply a chi-square measure to calculate how selectively words and phrases co-occur within the same sentences as a particular subset of frequent terms in the document text. The chi-square measure is applied to determine the bias of word co-occurrences in the document text which is then used to rank words and phrases as keywords of the document. Matsuo and Ishizuka (2004) state that the degree of biases is not reliable when term frequency is small. The authors present an evaluation on full text articles and a working example on a 27-page document, showing that their method operates effectively on large documents.

Another approach for keyword extraction is TextRank – a graph-based ranking model for text processing [Mihalcea, 2004]. In the graph, each word is presented as a node and the edges are constructed by observing the co-occurrence of words inside a moving window of predefined size between 2 and 10. TextRank is completely unsupervised and that makes it easily portable to other text collections, domains, and languages. Nevertheless, the authors report that TextRank achieves its best performance when only nouns and adjectives are selected as potential keywords.

In another work, instead of using only statistical data such as TF and n-grams, linguistic knowledge is also added to the equation. In detail, extracting NP-chunks gives a better precision than n-grams, and by adding the POS-tag(s) assigned to the term as a feature, a dramatic improvement of the results is obtained, independent of the term selection approach applied. [Hulth, 2003] In their paper, Rose et al. (2010) develop an unsupervised, domain-independent, and language-independent method for extracting keywords from individual documents. Rather than omitting the stopwords such as ‘the’, ‘a’, ‘of’, etc., they use them as a delimiter to split the text into candidate keywords. Next, a graph of co-occurrences is generated where each candidate is assigned a member word score, based on the degree and frequency of the vertices in the graph. This approach proved to be faster than TextRank while achieving higher precision and comparable recall scores.

YAKE! is Yet Another Keyword Extractor that relies on statistical text features extracted from individual documents to identify the most relevant keywords in the text. [Campos, 2020] YAKE! defines a set of five features capturing keyword characteristics which are heuristically combined to assign a single score to every keyword. The lower the score, the more significant the keyword will be. Experimental results carried out on top of twenty datasets have shown that YAKE! significantly outperforms other unsupervised methods on texts of different sizes, languages, and domains. After the Transformer came out in 2018, researchers tried incorporating it into many different natural language processing tasks, keyword extraction being one of them. In the following sections a Transformer-based approach will be presented. For a complete overview of the keyword extraction approaches, however, first some unsupervised automated keyword extraction models will be presented that combine the transformer architecture with simpler techniques. The Phraseformer is a model, that combines both Transformer and Graph Embedding techniques. For each keyword candidate, the text and the structure learning representation are concatenated and presented as a vector. Apart from that, the task is handled as a sequence labelling problem solved using classification task. The authors demonstrate that the combination of BERT and graph embeddings is more powerful that either of the approaches alone as it can better represent the semantic of the words and therefore outperforms the models using only one of the techniques.

Rungta et al. (2020) also explore the keyword extraction as a sequence labelling task to take advantage of the Transformer architecture and additionally use pre-trained weights to initialize the embedding layer of their model.

KeywordMap [Tu, 2021] also makes use of the attention scores provided by Transformer-based models and rely on links between words to build a keyword network. Then using a novel algorithm called Attention-based Word Influence (AWI), they manage to calculate the importance of each keyword candidate to the network. Moreover, KeywordMap, is developed to support multi-level analysis of keywords and keyword relationships through coordinated views.

By pretraining a Transformer-based language model on a domain-specific corpus and adapting its architecture for the task at hand, the TNT-KID model addresses the flaws of both supervised and unsupervised techniques for keyword extraction. [Martinc, 2021] This model outperforms existing models on numerous datasets, even though it only expects a small amount of training data.

## 3. Experiments

In this part the experimental set up will be presented and the results will be discussed. For each model, first a description if its architecture will be presented, then an example will be shown using the python package WordWise and only 300 observations for simplicity. Then, the code for the experiment will be presented and the results will be shown. Lastly, the evaluation will be summed up.

**Disclaimer:** The models take time to compute the whole dataset, which is longer than what Colab allows its users. For this reason, the whole dataset is computed separately on a CPU and only the final results are being presented here. The same code has been used as in the Example part, without the restriction to only compute the first 300 entries.

**Set up**

In order to compare the models, we need a common basis. For this experiment, the data set KP20k was chosen. [Meng, 2017] This data set consists of 20,000 abstracts of computer science papers, their titles and the keywords assigned by the authors. For consistence purposes, all models will be tested with the Pyton package WordWise. [Tae, 2021] At the end of this notebook, you will find another Python package that is useful for Keyword Extraction. Both packages can be used without additional pretraining and could be directly used.

**Evaluation Metric**

Before we dive into the different models, it is important to take a look at the evaluation and what would make one model better than another one. Undeniably, a good Keyword Extraction model is one, which is able to choose those words from a text, which best describe it. With the help of the data set, we have a common ground for the models, which would allow us to compare them. Intuitively, one could say that the better model is one that manages to predict more of the keywords of a text that were given by its author. As mentioned before, the model embeds the words from the text and then selects the ones with the smallest distance to the text embedding. This means that the models are only able to choose keywords that are already present in the text itself. In reality, authors do not always only choose keywords from the abstract, but usually opt for more general terms, contextual synonyms or words that describe the whole paper, which are simply not present in the abstract. Therefore, it is simply impossible for a Keyword Extraction model to show a high performance and to output all keywords given by the author. When evaluating models, one should always keep that in mind and expect lower performance scores.

Further limitation is number of keywords k that a model should produce. Different texts shall by nature correspond to different number of key phrases. [Yuan, 2020] But this aspect has not been widely considered in the past years when evaluating a Keyword Extraction model. Instead, a fixed number of keywords has been generated, typically 5 or 10, which resulted in attempts to match a fixed number of outputs against a variable number of true keywords. In the paper by Yuan, it was shown that for the several commonly used key phrase generation datasets the average number of key phrases per data point can range from 5.3 to 15.7, with variances sometimes as large as 64.6, which resulted in F1 score for the oracle model on the KP20K dataset being 0.858 for k = 5 and 0.626 for k = 10. The number of generated key phrases used for evaluation can have a critical impact on the quality of the resulting evaluation metrics. Therefore, as suggested by the authors of the paper, here we will be using a variable length of the key phrases. Namely, for each abstract the models will produce as many keywords as the number of the ground truth keywords.

However, another aspect to consider is the order of the keywords proposed by the model. The model outputs the top k words, whose embeddings are as close to the embedding of the text, which means that the first suggested keyword is the closest and the distance increases with the following suggestions. Therefore, only comparing the predicted keywords to the true keywords would be not entirely correct. Yuen et.al. propose to change the order of the ground truth keywords by sorting them by their order of occurrence and to append the key phrases that do not appear in the text to the end. Then they use a variable number of predicted keywords to calculate the F1 score. This is an interesting workaround, but there are other limitations to both the approach and the F1 score. Imagine the simplified situation in which a text has 5 keywords. Two models A and B are being examined and both have only one correct prediction and it is the same keyword. Now model A predicts it at the first place, i.e. with closest representation to the text and model B puts it on 5th position. If we only consider whether the predicted keyword is present in the set of ground truth keywords, this would mean that both models performed equally well. However, model A was able to capture the importance of this keyword better and extracts relevant key phrases higher up the order as compared to model B. Therefore, we need a metric that is able to capture and evaluate this behaviour as well.

A metric like that is the Mean Average Precision (MAP). It is a metric that penalizes if irrelevant key-phrases are extracted higher up the order and gradually decrease the significance of the errors (extraction of irrelevant key-phrase) as we go down the list of extracted key-phrases. [Shrivastava, 2020] The MAP is calculated as follows: for each predicted keyword the precision of the list until this keyword is calculated. Next, the average sub-list precision scores are computed. Finally, the mean average precision for all documents is calculated.

When comapring the predicted keywords to the ground truth, it is imortant that both lists are stemmed. For the case when an author has given the phrase 'NLP models' as a key segment, but the model predicts 'NLP model', a not stemmed output would be evaluated as incorrect. If both lists are stemmed, however, then they both would be in singular form and the prediction would be correct.

In this notebook, the Poter Stemming will be used.


**Main Steps**
- Load in packages and the dataset
- Models: For each of the models a short description is provided as well the necessary code for the experiments
    - BERT
    - RoBERTa
    - DistilBERT
    - DistilRoBERTa

## 4. Empirical Results

The results of the experiments can be seen n the table below. In the first row the MAP score of each model is presented and in the second row the time it took for the model to process the data.

Test | BERT | Roberta | DistilBERT | DistilRoBERTa
--- |--- | ---|--- | ---
MAP score | 0.122 |	0.113 |	0.144 |	0.118
Time | 4:16:04 |	4:40:54 |	2:16:03 |	2:18:02

DistilBERT achieved the highest score for Keyword Extraction for the data set KP20K and it finished the task for the shortest time. Overall, all models performed similarly well on this data set. In the literature, it was experimented a lot with different datasets and some models perform well on some, while others perform better on other datasets.

When choosing a language model, it is important to consider all aspects of its architecture as well as its computational time and resources.

**Additional Python Package**

**KeyBERT**

Another package that can be used for Keyword Extraction is KeyBERT. The package WordWise has been inspired by KeyBERT. KeyBERT has additional parameters to set the dissimilarity technique (MMR or MSS and its level of diversity). Furthermore, it has the nice additional feature to output the text for which the keywords are searched and highlight them. Depending on the fine-tuning of the additional parameters, KeyBERT can deliver very different results and therefore MAP scores. For simplicity reason, the package WordWise was chosen to be able to compare all language models on the same level. Moreover, KeyBERT takes significantly more time to compute the results than WordWise.

## 6. Conclusion

In this paper the downstream task keyword extraction was analysed. The main approach of this task was described together with some tips on how to get more heterogeneous results. The different ways on how to embed words and text using different models were presented. The language models BERT, RoBERTa, DistilBERT and DistilRoBERTa were described and the main differences between them were highlighted. Further, the performance of the models was compared and the model with the highest MAP score and shortest computation tome was indicated. The downsides of other measuring techniques were discussed. Lastly, python packages for keyword extraction were presented.

## References

[Bucila, 2006]

    Bucila, Cristian, Caruana, Rich, and Niculescu-Mizil, Alexandru. Model compression. In KDD, 2006.

[Campos, 2019]

    Campos, Ricardo, et al. “YAKE! Keyword Extraction from Single Documents Using Multiple Local Features.” Information Sciences, vol. 509, 1 Jan. 2020, pp. 257–289, www.sciencedirect.com/science/article/abs/pii/S0020025519308588?casa_token=NZfi-GmbK1gAAAAA:WC5tKprHpm5fy2askOGZsc_sFyqklbjNqUGrb7ipJZLTwgzqlPem_tqDDjy_rL_u44w2X_VVkkQ, 10.1016/j.ins.2019.09.013.

[Carbonnel, 1998]

    Carbonell, Jaime. The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. 1998.

[Devlin, 2019]

    Devlin, Jacob, et al. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. 2019.

[Hinton, 2015]

    Hinton, Geoffrey E. , Vinyals, Oriol , and Dean, Jeffrey . Distilling the knowledge in a neural network. ArXiv, abs/1503.02531, 2015.

[Gokaslan, 2019]

    Gokaslan, Aaron and Cohen, Vanya. 2019. Openwebtext corpus. http://web.archive.org/ save/http://Skylion007.github.io/ OpenWebTextCorpus.

[Grootendorst, 2020]

    Grootendorst, Maarten. “Home - KeyBERT.” Maartengr.github.io, 29 Oct. 2020, maartengr.github.io/KeyBERT/index.html.

[Hulth, 2003]

    Hulth, Anette. Improved Automatic Keyword Extraction given More Linguistic Knowledge. June 2003. Department of Computer and Systems SciencesStockholm University.

[Liu, 2019]

    Liu, Yinhan, et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. 2019.

[Martinc, 2021]

    Martinc, Matej, et al. “TNT-KID: Transformer-Based Neural Tagger for Keyword Identification.” Natural Language Engineering, 10 June 2021, pp. 1–40, www.cambridge.org/core/journals/natural-language-engineering/article/tntkid-transformerbased-neural-tagger-for-keyword-identification/A41C8B12C1F3F4F02BF839FCAFA1A695, 10.1017/S1351324921000127.

[Meng, 2017]

    Meng, R., Zhao, S., Han, S., He, D., Brusilovsky, P., and Chi, Y. (2017). Deep keyphrase generation. In ACL.

[Mihalcea, 2004]

    Mihalcea, Rada, and Paul Tarau. TextRank: Bringing Order into Texts. 2004. Nikzad-Khasmakhi, N., et al. “Phraseformer: Multimodal Representation Learning for Expert Recommendation System with Transformers and Graph Embeddings.” Chaos, Solitons & Fractals, vol. 151, Oct. 2021, p. 111260, arxiv.org/pdf/2106.04939.pdf, 10.1016/j.chaos.2021.111260.

[Nagel, 2016]

    Nagel, Sebastian 2016. Cc-news. http: //web.archive.org/save/http: //commoncrawl.org/2016/10/newsdataset-available.

[Persson, 2021]

    Persson, Sanna. “Paper Summary — BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” Analytics Vidhya, 25 May 2021, medium.com/analytics-vidhya/paper-summary-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-861456fed1f9. Accessed 16 Jan. 2022.

[Rose, 2021]

    Rose, Stuart, et al. “Download Limit Exceeded.” Citeseerx.ist.psu.edu, 2010, citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.657.8134&rep=rep1&type=pdf. Accessed 23 Nov. 2021.

[Rungta, 2020]

    Rungta, Mukund, et al. “TransKP: Transformer Based Key-Phrase Extraction.” IEEE Xplore, 1 July 2020, ieeexplore.ieee.org/abstract/document/9206812.

[Sanh, 2019]

    Sanh, Victor. “🏎 Smaller, Faster, Cheaper, Lighter: Introducing DilBERT, a Distilled Version of BERT.” Medium, HuggingFace, 28 Aug. 2019, medium.com/huggingface/distilbert-8cf3380435b5.

[Sanh, 2020]

    Sanh, Victor, et al. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. 1 Mar. 2020.

[Shrivastava, 2020]

    Shrivastava, Ishan. “Exploring Different Keyword Extractors — Evaluation Metrics and Strategies.” GumGum Tech Blog, 21 July 2020, medium.com/gumgum-tech/exploring-different-keyword-extractors-evaluation-metrics-and-strategies-ef874d336773.

[Singh, 2021]

    Singh, Aastha. “Evolving with BERT: Introduction to RoBERTa.” Analytics Vidhya, 9 July 2021, medium.com/analytics-vidhya/evolving-with-bert-introduction-to-roberta-5174ec0e7c82.

[Trieu, 2018]

    Trieu H Trinh and Quoc V Le. 2018. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847.

[Tae, 2021]

    Tae, Jake. “WordWise.” GitHub, 28 Jan. 2022, github.com/jaketae/wordwise.

[Tu, 2021]

    Tu, Yamei, et al. “KeywordMap: Attention-Based Visual Exploration for Keyword Analysis.” IEEE Xplore, 1 Apr. 2021, ieeexplore.ieee.org/abstract/document/9438768.

[Wasserblat, 2021]

    Wasserblat, Moshe. “Best Practices for Text Classification with Distillation (Part 1/4) - How to Achieve BERT Results by Using Tiny Models.” Www.linkedin.com, 17 May 2021, www.linkedin.com/pulse/best-practices-text-classification-distillation-part-14-wasserblat.

[Yuan, 2020]

    Yuan, Xingdi, et al. One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases. Association for Computational Linguistics, 2020.

[Zhu, 2015]

    Zhu, Yukun, et al. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. 2015, In Proceedings of the IEEE international conference on computer vision, pages 19–27.
