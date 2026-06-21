## Research and Applications

# Utilizing active learning strategies in machine-assisted annotation for clinical named entity recognition: a comprehensive analysis considering annotation costs and target effectiveness

Jiaxing Liu , PhD1 and Zoie S.Y. Wong, PhD�,2,3,4 

1 School of Statistics and Mathematics, Zhongnan University of Economics and Law, Wuhan, Hubei 430073, China, 2 Graduate School of Public Health, St Luke’s International University, OMURA Susumu & Mieko Memorial St Luke’s Center for Clinical Academia, Chuo-ku, Tokyo 104-0045, Japan, 3 The Kirby Institute, University of New South Wales, Sydney, NSW 2052, Australia, 4 School of Medical Sciences, The Unviersity of Sydney, Camperdown, NSW 2050, Australia 

�Corresponding author: Zoie S.Y. Wong, PhD, Graduate School of Public Health, St Luke’s International University, 3-6-2 Tsukiji, Chuo-ku, Tokyo 104-0045, Japan (zoiesywong@gmail.com) 

## Abstract

Objectives: Active learning (AL) has rarely integrated diversity-based and uncertainty-based strategies into a dynamic sampling framework for clinical named entity recognition (NER). Machine-assisted annotation is becoming popular for creating gold-standard labels. This study investigated the effectiveness of dynamic AL strategies under simulated machine-assisted annotation scenarios for clinical NER. 

Materials and Methods: We proposed 3 new AL strategies: a diversity-based strategy (CLUSTER) based on Sentence-BERT and 2 dynamic strategies (CLC and CNBSE) capable of switching from diversity-based to uncertainty-based strategies. Using BioClinicalBERT as the foundational NER model, we conducted simulation experiments on 3 medication-related clinical NER datasets independently: i2b2 2009, n2c2 2018 (Track 2), and MADE 1.0. We compared the proposed strategies with uncertainty-based (LC and NBSE) and passive-learning (RANDOM) strategies. Performance was primarily measured by the number of edits made by the annotators to achieve a desired target effectiveness evaluated on independent test sets. 

Results: When aiming for 98% overall target effectiveness, on average, CLUSTER required the fewest edits. When aiming for 99% overall target effectiveness, CNBSE required 20.4% fewer edits than NBSE did. CLUSTER and RANDOM could not achieve such a high target under the pool-based simulation experiment. For high-difficulty entities, CNBSE required 22.5% fewer edits than NBSE to achieve 99% target effectiveness, whereas neither CLUSTER nor RANDOM achieved 93% target effectiveness. 

Discussion and Conclusion: When the target effectiveness was set high, the proposed dynamic strategy CNBSE exhibited both strong learning capabilities and low annotation costs in machine-assisted annotation. CLUSTER required the fewest edits when the target effectiveness was set low. 

Key words: active learning; named entity recognition; clinical notes; medication extraction; pre-trained language model. 

## Introduction

Clinical named entity recognition (NER) has been recognized as a fundamental method to extract meaningful clinical concepts from clinical notes, for example, medication information, adverse drug events (ADEs), etc.1,2 Traditionally, annotating gold-standard named entities (NEs) required skilled annotators with expertise in the relevant discipline, consequently, incurring substantial labor costs. Nowadays, this human-driven labeling process can be assisted by machine pre-annotation.3,4 For example, Todd Lingren et al reported that machine pre-annotation could save between 13.9% and 21.5% of annotation time when annotating clinical NEs such as medication, symptom, procedure, disease, medication type, and anatomical site in a corpus of clinical trial announcements.4 A recent study demonstrated that machine annotation using ChatGPT can achieve 59%-83% accuracy on text annotation tasks in general domain corpora such as tweets and news articles, outperforming MTurk crowd workers.5 Studies have revealed that zero-shot or few-shot NER using GPT models achieved 63.4%-86.1% F1 scores for identifying medical problems, treatments, and tests from synthetic clinical notes6 ; additionally, F1 scores regarding identifying nervous system disorder-related adverse events from safety reports were 30.1%-73.6%.6,7 Despite the prominent developments regarding pre-trained language models for clinical-concept extraction,8 skilled labor remains essential for gold-standard annotation, since accurate labeling is crucial for fine-tuning accurate downstream tasks. 

Active learning (AL), a machine learning technique that actively queries humans while labeling data, is designed to minimize the amount of data review required for model training. This approach iteratively selects the most informative samples for human labeling to maximize model performance. AL methods have been effective in a wide range of applications, including image classification,9,10 text classification,11– 15 and NER.16,17 Existing studies have evaluated AL strategies with NER models in CRF or BiLSTM. Transfer learning using pre-trained language models requires fewer data for generalization. When integrated with AL for annotation, fine-tuning pre-trained language models have the potential to substantially enhance performance and effectiveness. Currently, only a few studies have evaluated the performance of AL incorporated with pre-trained language models.16–18 Shelmanov et al18 proposed an uncertainty-based AL strategy integrated with Bio-BERT, which demonstrated that fewer AL iterations can achieve satisfactory performance compared with that of random selection, a passive-learning strategy. Selecting informative instances via CRF input and output, Liu et al17 demonstrated improved NER performance via AL by the lowest token probability using bidirectional encoder representations from transformers (BERT), as compared with that of traditional strategies. Furthermore, AL with machine-assisted annotation can potentially further reduce human effort in clinical NER tasks, a potential that is rarely explored in existing studies, except for Kholghi et al. 19 

In terms of AL sampling approaches, Chen et al20 examined AL strategies based on uncertainty and diversity samplings and compared them with random selection. Uncertainty-based sampling tends to select the most uncertain sentence based on the predicted probability or entropy of sentences21–23 or tokens.17,18,24 Diversity-based sampling groups semantically similar sentences through sentence representation and sentence clustering.20,25 For example, sentences covering medications in similar contexts, such as during prescription, could be grouped into the same cluster. Therefore, annotating 1 sentence (eg, with the prescribed medication “Dulcolax”) could enable the NER model to identify other prescribed medications (“Amaryl,” “Nortriptyline,” etc.). Furthermore, these AL strategies select only a few sentences per cluster, avoiding redundant annotation of semantically similar sentences. Some combined AL strategies have also been proposed in past studies, integrating both uncertaintyand diversity-based strategies.25,26 For example, Kholghi et al25 proposed a new group of two-level hybrid AL strategies called Clustering And Representation Learning Sampling (CARLS): in this approach, first, the sentences are clustered based on sentence representation; then the sentences in clusters are selected based on either the least uncertainty scores, their dissimilarity to the labeled set, or a blend of both. Other AL strategies incorporate medical knowledge, rules, annotation costs, and other metrics.27,28 To the best of our knowledge, all existing AL strategies remain unchanged throughout the iterative selection process. However, as the annotated data expand and the NER model learns, employing diverse AL strategies with varying selective strengths may facilitate better data selection, thereby improving NER model performance at different stages. This could lead to pre-trained language models being fine-tuned to achieve higher performance at reduced annotation costs. 

This study aims to propose new AL strategies and examine their effectiveness in reducing annotation costs. These evaluations are conducted under the growingly popular machineassisted annotation scenarios, where human annotators review pre-annotations from models iteratively to complete the labeling process. We examined the proposed AL strategies using 3 gold-standard clinical NER datasets and evaluated their performance using a set of fair annotation metrics appropriate for machine-assisted annotation contexts. Inspired by Liu et al,17 we also considered different target effectiveness levels to compare AL performances at different desired achievement levels. In this study, we also introduced a novel measure of annotation cost aimed at evaluating the efforts of human annotators in correcting the machine preannotations. 

## Methods

## Datasets

In this study, we used 3 clinical NER datasets that contained gold-standard labels: i2b2 2009 dataset,29 n2c2 2018 (Track 2) dataset,1 and MADE 1.0 dataset.30 We provide a brief description of the datasets below. 

1) i2b2 2009 contains 1243 de-identified discharge summaries but only 261 reports with gold-standard annotations. It identifies 6 entity types: medication, dosage, mode (route) of administration, frequency, duration, and reason for administration. 

2) n2c2 2018 (Track 2) consists of 505 discharge summaries drawn from the MIMIC-III clinical care database; 303 files were designated for training and 202 files were held out for testing. It has 9 entity types, including drug, strength, form, dosage, frequency, route, duration, reason, and ADE. 

3) MADE 1.0 is comprised of 1089 de-identified clinical notes including discharge summaries, consultation reports, etc., 876 of which were used for training and 213 used for testing. The dataset has 9 entity types: medication, indication, frequency, severity, dosage, duration, route, ADE, and SSLIF (any sign, symptom, and disease that is not an ADE or indication). 

Our study requested access to the datasets and ensured compliance with all relevant data usage policies. The datasets were all provided in a de-identified format by their respective providers. Our research did not involve direct interaction with human subjects. After careful consideration, it was determined that our study did not require Institutional Review Board approval. 

In our experiment, we used the original training and testing splits provided by n2c2 2018 and the MADE 1.0 corpus. As for i2b2 2009, we randomly selected 80% of reports (208) to be used for training and 20% of reports (53) to be used for testing. Datasets are tokenized, preprocessed, and tagged using a standard BIO format, where “B” and “I” mark the “beginning” and “inside” of an NE; “O” indicates the token is “outside” of any entity (Note: Figure S1 shows a sample sentence, its tokens, identified entities, and their representation in BIO format in a sample discharge summary report taken from i2b2 200929) NER tasks are performed at the sentence level. We summarize the descriptive statistics and the distribution of entity types of the 3 corpora in Tables S1-S4. 

## NER model

BERT is a state-of-the-art pre-trained language model built on multi-layer transformers.31 It was pre-trained on large amounts of text data, allowing it to capture rich representations of text data. BioClinicalBERT8 is a BERTbased model that has been pre-trained on a large corpus of MIMIC-III v1.4 data and has proven to be more suitable for clinical natural language processing. In this study, we used 3 separate datasets to fine-tune BioClinicalBERT to perform AL simulation experiments for clinical NER. 

When fine-tuning the BioClinicalBERT model, the parameters were optimized using the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 5 }$ . The batch size was 32 in training, evaluation, and testing and the epoch of training was 5. The maximum input sentence length was set to 128. Sentences longer than the maximum length were broken into 2 or more sentences. The default hyperparameters were used, as defined in Alsentzer et al8 

## Experiment design

Referring to the established experimental designs,16,17 we utilized a pool-based training framework32 that iteratively evaluates and ranks the entire pool of the unlabeled dataset, selecting the most informative query samples to train the NER model until meeting the stopping criteria. Figure 1 describes the experiment workflow. The experiments were performed independently on 3 datasets. Therefore, we iteratively fine-tuned separate models for each dataset. We used the percentage-based selection to account for the differing overall sizes across different datasets. The experiment started with a small dataset (1% of the total sentences from each training dataset) randomly selected from the unlabeled dataset and annotated to train an initial NER model. Next, we examined different AL strategies to score the unlabeled data and selected a batch of informative instances for annotation (1% of the total tokens from each training dataset for each iteration, ie, 2469 tokens for i2b2 2019, 8951 tokens for n2c2 2018, and 7810 tokens for MADE 1.0). Then, the NER model was retrained from scratch during each iteration. The trained NER model was also used to pre-annotate the unlabeled data for human annotation in the next iteration. The procedure was repeated until either the desired level of performance was attained (a 98% or 99% micro-averaged F1 score comparable to the performance achieved using the entire training dataset) or the annotation budget was exhausted (30 iterations). 

During the experiment, we masked the labels of the training datasets, initially treating them as unlabeled. The labels were gradually added as the experiment proceeded to simulate the human annotation. Using different random seeds, we repeated the entire experiment 3 times for each of the 3 datasets, which resulted in differing initial sets of data and their subsequent selections. The hold-out testing sets were used to evaluate the effectiveness of different AL strategies externally. 

## AL strategies

In this study, we proposed 3 AL strategies, listed below as (4- $^ { 6 ) , }$ and compared them with (1-3). We describe these strategies below: 

1) RANDOM was the baseline passive-learning strategy that selects sentences randomly in each query iteration. 

2) Least Confidence (LC),21 which is an uncertainty-based approach, selects sentences that the NER model is least confident about for annotation to reduce the model’s uncertainty and improve its performance. 

3) N-best sequence entropy (NBSE),33 which is an uncertainty-based approach, calculates the entropy of probability distribution over N-best possible sentence labels predicted by the NER model and selects the sentences with the highest entropy. We referred to Kim et al33 and used the same default setting N ¼ 3. 

4) We proposed a new diversity-based strategy, called CLUS-TER, which selects sentences from each cluster to ensure query diversity in each iteration. As a diversity-based AL strategy has rarely been used alongside BERT, we pioneered Sentence-BERT34 to encode the sentences and group them into K (K ¼ 100) clusters using the K-means algorithm. CLUSTER selects N/K instances from each 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-05/5815a139-d2ff-4eff-910e-31828ac64fc2/13c63bfbe25ae1dcbd73df5bacc4740a50c9b68615977b596c61f20c483e9afa.jpg)



Figure 1. The pool-based active learning workflow and experiment design.


cluster to form the query batch. The number of clusters was set to 100 based on the results from a preliminary experiment comparing K ¼ 50, 100, and 150 shown in Table S5. 

5) We newly proposed a dynamic strategy, called CLC. It utilizes CLUSTER for the first few iterations until the reduction in training loss compared to the previous iteration is less than 0.005. Then, CLUSTER switches to the uncertainty-based strategy LC. The loss function is the Cross-Entropy loss, which can be defined as follows: 

$$
C E (y, \widehat {y}) = - \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {C} y _ {i j} \log \left(\widehat {y} _ {i j}\right)
$$

where N is the sentence length, C is the total number of possible classes, $y _ { i j }$ indicates whether token i belongs to class $j ,$ and $\widehat { y } _ { i j }$ is the predicted probability that token i belongs to class j. 

6) We proposed another new dynamic strategy, namely CNBSE. It is identical to (5) dynamic strategy CLC, except CLUSTER switches to the uncertainty-based strategy NBSE when the reduction in training loss compared to the previous iteration is less than 0.005. 

## Annotation cost

In the simulation study, it is crucial to identify an appropriate annotation cost metric to estimate the annotation effort and evaluate the effectiveness of AL strategies. Some studies are measuring the number of AL iterations,17,18,25,35 the number of sentences,23,24,28,35 and the number of tokens20,22,26,28 to evaluate annotation effort. However, in reality, the amount of effort spent to annotate each token, sentence, or AL iteration is not the same.19,26 For example, longer sentences take more time to read; the number of entities or length of entities within a sentence also impacts annotation time.19 Thus, the aforementioned metrics of annotation effort are inadequate for assessing the actual annotation cost borne by human annotators. 

We therefore introduce a new annotation cost metric, the number of edits, as the primary measure. The number of edits refers to the minimum number of changes (including insertions, deletions, and replacements) necessarily made by human annotators to correct the machine’s pre-annotation, as illustrated in Figure 2. We quantify the number of edits using the Levenshtein distance.36 This is considered a fair measure for machine-assisted annotation as annotators establish gold-standard labels by rectifying incorrect preannotations, eliminating excessive pre-annotations, and introducing missing annotations. Further information about the Levenshtein distance can be found in the Supplementary Material. Referring to Kholghi et al19 and Wei et al,27 we also adopted 4 annotation cost metrics as secondary measures. These are the number of sentences, the number of tokens, the number of entities, and the number of entity tokens. Altogether, we used 5 annotation cost metrics for our evaluation. 

## Evaluation metrics for AL strategies: annotation rate

We evaluated the effectiveness of AL strategies by measuring the reductions in annotation effort across the 5 annotation cost metrics, using the annotation rate as described by Kholghi et al19,37 The annotation rate was computed for each annotation cost metric as follows: 

Annotation Rate 

$$
= \frac {\# \text { annotation   cost   used   by   AL   strategy }}{\# \text { annotation   cost   using   the   entire   training   data }},
$$

indicating the efficiency of AL strategies in achieving the target effectiveness compared with the highest effectiveness using the entire training data. A lower annotation rate, therefore, indicates a more effective AL strategy. To compute annotation rates, as indicated in Table S1, we used the counts of sentences, tokens, entities, and entity tokens within the entire training datasets as the denominators for each corresponding metric. Regarding the denominator for the number of edits annotation rate, we used the count of entities, assuming that all entity tags need to be added by the annotators. 

Furthermore, we used micro-averaged F1 scores to assess the overall performance of the AL-incorporated NER models. For the n2c2 2018 dataset, we used the evaluation scripts provided by the original challenges to calculate the lenient F1 measures.1 For the MADE 1.0 dataset and i2b2 2009 dataset, we use the Python seqeval package38 for evaluation. The highest F1 scores reached by complete training data and evaluated against the testing set were used to set the 100% overall target effectiveness that AL strategies aimed to achieve. The micro-averaged F1 scores were 0.920 for i2b2 2009, 0.917 for n2c2 2018, and 0.830 MADE 1.0. The entity-level performances for the 3 corpora are presented in Tables S6- S8. These set the overall and by-entity 100% target effectiveness of AL strategies. Ideally, AL strategies should be able to attain 100% target effectiveness, or even surpass it within 30 iterations (equivalent to utilizing around 31% of tokens from the complete training dataset). However, the ideal threshold might not always be able to achieve. We set 2 other target effectiveness thresholds, 98% and 99%. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-05/5815a139-d2ff-4eff-910e-31828ac64fc2/ca464f0ba7aa7e59a8ddf99e6cb6b704bf70b9f3a34205c93a3e3de3158f1056.jpg)



Figure 2. An illustration of the number of edits. In this case, the number of edits is 3 because it requires 1 insertion, 1 replacement, and 1 deletion to amend the machine pre-annotation to the gold-standard level.


We also conducted a named entity-based, sub-group analysis. Two typical types of NEs, based on their identification difficulty using the NER model under the 100% overall target effectiveness setting (trained using complete training data and evaluated against the test set) were defined as follows: 

� Low-difficulty group: for each dataset, we select the top 2 NEs with the highest F1 scores. 

� High-difficulty group: for each dataset, we select the bottom 2 NEs with the lowest F1 scores. 

## Results

## Annotation rates comparison based on target effectiveness across different AL strategies

The AL experiments compared 6 AL strategies, and the performances were tested on the hold-out testing sets. Each set of experiments was conducted 3 times and the mean annotation rates by annotation-cost metric to achieve 98% or 99% target effectiveness, as shown in Table 1. 

Under the machine-assisted annotation scenario, as shown in Table 1, CLUSTER achieved a 98% target effectiveness with the lowest annotation rates for edits across all the datasets, presenting an overall mean value of 4.0%. The strategy performance evaluation depends on the selected annotationcost metric considered. For 98% target effectiveness, NBSE displayed low annotation rates for tokens, whereas CLUS-TER demonstrated low annotation rates for entities. On average, all AL strategies in CLUSTER, LC, NBSE, CLC, and CNBSE reduced the number of tokens to be reviewed compared to RANDOM, with reductions reported as 19.3%, 55.2%, 59.2%, 54.3%, and 58.5%, respectively. 

Both CLUSTER and RANDOM failed to reach a higher target effectiveness, 99%, within 30 iterations in 3 repeated experiments on the 3 datasets, except for 1 experiment in CLUSTER using i2b2 2009. However, uncertainty-based strategies (LC and NBSE) and dynamic strategies (CLC, CNBSE) could consistently achieve a 99% target effectiveness on the 3 datasets. As shown in Table 1, CNBSE required the smallest annotation rate for edits. With the incorporation of CLUSTER into uncertainty-based strategies in the dynamic AL strategy, CLC reduced an average percentage of 16.9% edits compared to LC and CNBSE reduced an average percentage of 20.4% edits compared to NBSE. However, on average, CNBSE required 64.0% more edits to achieve a 99% target effectiveness compared with the edits needed to achieve a 98% target effectiveness. The numbers of insertions, deletions, and replacements required to achieve a 99% target effectiveness are presented in Table S9. Dynamic AL strategies consistently reduced the numbers of insertions, deletions, and replacements compared to uncertainty-based strategies; necessary edits consisted primarily of insertions. 

The learning curves expressing micro-averaged F1 scores vs the number of edits on the 3 datasets are shown in Figure 3. The horizontal lines represent 100%, 99%, and 98% target effectiveness of micro-averaged F1 scores, respectively. Figures S2-S4 show more comprehensive plots, presenting 5 annotation-cost metrics and the number of AL iterations. 

As displayed in Figure 3, dynamic strategies (CLC and CNBSE) outperformed the other strategies, reaching a 99% or 100% target effectiveness with fewer edits. However, when considering a 98% target effectiveness, on average, the numbers of edits required by uncertainty-based strategies (LC and NBSE) were 184.6% and 146.8% larger than CLUSTER. In comparison, dynamic strategies also required more edits (CLC: 106.9%; CNBSE: 57.2%) than CLUSTER but in a smaller magnitude than the uncertainty-based strategies. Similarly, uncertainty-based strategies required more entities to be reviewed (437.7% for LC and 392.4% for NBSE) and more edits to be made (227.2% for LC and 218.3% for NBSE) compared to CLUSTER in the first 3 iterations, shown in Figures S5 and S6. 

To account for the potential variations in token size, we have also experimented on a fixed number of token selection in each AL iteration and a similar conclusion persists. Dynamic strategy CNBSE has lower average annotation rates for edits, entities, and tokens when aiming at high target effectiveness on the 3 datasets, see Tables S10-S12. 


Table 1. Mean Annotation rates for different AL strategies to attain 98% or 99% target effectiveness across 3 datasets.


<table><tr><td rowspan="2">Annotation rates</td><td rowspan="2">AL strategies</td><td colspan="3">98% Target effectiveness</td><td colspan="3">99% Target effectiveness</td></tr><tr><td>i2b2 2009 (%)</td><td>n2c2 2018 (%)</td><td>MADE 1.0 (%)</td><td>i2b2 2009 (%)</td><td>n2c2 2018 (%)</td><td>MADE 1.0 (%)</td></tr><tr><td rowspan="6">Edits</td><td>RANDOM</td><td>5.2</td><td>3.6</td><td>5.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CLUSTER</td><td>4.6</td><td>2.8</td><td>4.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LC</td><td>13.3</td><td>9.4</td><td>10.3</td><td>16.6</td><td>11.6</td><td>12.3</td></tr><tr><td>NBSE</td><td>13.7</td><td>7.6</td><td>7.7</td><td>18.3</td><td>10.2</td><td>10.3</td></tr><tr><td>CLC</td><td>9.1</td><td>7.3</td><td>7.3</td><td>13.4</td><td>10.4</td><td>9.7</td></tr><tr><td>CNBSE</td><td>7.6</td><td>4.6</td><td>6.4</td><td>11.4</td><td>9.7</td><td>8.4</td></tr><tr><td rowspan="6">Tokens</td><td>RANDOM</td><td>21.2</td><td>13.7</td><td>20.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CLUSTER</td><td>16.5</td><td>10.4</td><td>18.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LC</td><td>10.8</td><td>5.3</td><td>9.3</td><td>16.2</td><td>7.3</td><td>11.7</td></tr><tr><td>NBSE</td><td>10.0</td><td>5.0</td><td>8.0</td><td>17.1</td><td>7.3</td><td>11.3</td></tr><tr><td>CLC</td><td>10.1</td><td>6.3</td><td>9.0</td><td>16.2</td><td>9.4</td><td>12.0</td></tr><tr><td>CNBSE</td><td>9.4</td><td>5.3</td><td>8.6</td><td>15.8</td><td>10.0</td><td>11.3</td></tr><tr><td rowspan="6">Entities</td><td>RANDOM</td><td>20.6</td><td>13.8</td><td>20.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CLUSTER</td><td>17.0</td><td>9.3</td><td>19.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LC</td><td>33.0</td><td>20.5</td><td>18.0</td><td>50.4</td><td>27.2</td><td>21.6</td></tr><tr><td>NBSE</td><td>34.2</td><td>17.6</td><td>16.3</td><td>55.2</td><td>27.1</td><td>22.3</td></tr><tr><td>CLC</td><td>28.4</td><td>19.5</td><td>15.9</td><td>45.6</td><td>31.4</td><td>20.4</td></tr><tr><td>CNBSE</td><td>24.9</td><td>14.4</td><td>15.2</td><td>44.9</td><td>34.0</td><td>19.9</td></tr></table>


RANDOM and CLUSTER failed to achieve a 99% target effectiveness. The lowest annotation rates of the AL strategies to achieve 98% or 99% target effectiveness in each of the 3 datasets are marked in bold. 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-05/5815a139-d2ff-4eff-910e-31828ac64fc2/92e17ce0d78dfe91880b9e65ca807be53088056e435689d3bab4026ddf77fb20.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-05/5815a139-d2ff-4eff-910e-31828ac64fc2/de1da85c03a30aaea0788465d26a26b7844269862af02d8413d5f8f925709982.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-05/5815a139-d2ff-4eff-910e-31828ac64fc2/8a9649a6979df4b65080b0a6ad8a065beebad43254f9acfd3dce80d67b597b17.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-05/5815a139-d2ff-4eff-910e-31828ac64fc2/02a0ac546f54cee6ad9f6181f3bcb8cf3ab043aca335ae998f0aafe058a11ffc.jpg)



Figure 3. Learning curves of micro-averaged F1 versus the number of edits.



Table 2. Mean annotation rates for different AL strategies to achieve a 99% target effectiveness for low-difficulty entities across 3 datasets.


<table><tr><td rowspan="2">Annotation rate</td><td rowspan="2">AL strategy</td><td colspan="2">i2b2 2009</td><td colspan="2">n2c2 2018</td><td colspan="2">MADE 1.0</td></tr><tr><td>Mode (%)</td><td>Dosage (%)</td><td>Strength (%)</td><td>Form (%)</td><td>Route (%)</td><td>Drug (%)</td></tr><tr><td rowspan="6">Edits</td><td>RANDOM</td><td>5.1</td><td>5.1</td><td>2.4</td><td>5.3</td><td><eq>7.4^a</eq></td><td>4.1</td></tr><tr><td>CLUSTER</td><td>4.1</td><td>4.2</td><td>2.2</td><td>6.0</td><td><eq>5.9^a</eq></td><td>3.3</td></tr><tr><td>LC</td><td>13.0</td><td>11.6</td><td>6.8</td><td>11.6</td><td>14.6</td><td>9.4</td></tr><tr><td>NBSE</td><td>14.6</td><td>12.0</td><td>5.9</td><td>12.5</td><td><eq>11.9^a</eq></td><td>6.2</td></tr><tr><td>CLC</td><td>9.3</td><td>8.0</td><td>4.0</td><td>11.1</td><td><eq>14.9^a</eq></td><td>6.8</td></tr><tr><td>CNBSE</td><td>7.7</td><td>8.4</td><td>2.6</td><td>8.7</td><td>14.2</td><td>6.0</td></tr><tr><td rowspan="6">Tokens</td><td>RANDOM</td><td>20.9</td><td>20.9</td><td>8.7</td><td>20.7</td><td><eq>31.0^a</eq></td><td>15.3</td></tr><tr><td>CLUSTER</td><td>14.1</td><td>14.8</td><td>8.3</td><td>23.7</td><td><eq>24.0^a</eq></td><td>12.3</td></tr><tr><td>LC</td><td>10.5</td><td>8.5</td><td>3.3</td><td>7.3</td><td>15.0</td><td>8.3</td></tr><tr><td>NBSE</td><td>11.4</td><td>7.6</td><td>3.7</td><td>9.7</td><td><eq>14.0^a</eq></td><td>6.3</td></tr><tr><td>CLC</td><td>10.5</td><td>8.5</td><td>4.3</td><td>10.0</td><td><eq>19.0^a</eq></td><td>8.7</td></tr><tr><td>CNBSE</td><td>9.8</td><td>10.4</td><td>4.0</td><td>9.3</td><td>20.3</td><td>8.3</td></tr><tr><td rowspan="6">Entities</td><td>RANDOM</td><td>20.4</td><td>20.3</td><td>8.7</td><td>20.7</td><td><eq>31.5^a</eq></td><td>15.4</td></tr><tr><td>CLUSTER</td><td>14.6</td><td>15.3</td><td>7.3</td><td>21.1</td><td><eq>26.3^a</eq></td><td>13.3</td></tr><tr><td>LC</td><td>32.3</td><td>26.1</td><td>14.0</td><td>27.2</td><td>26.4</td><td>16.7</td></tr><tr><td>NBSE</td><td>38.1</td><td>26.8</td><td>12.8</td><td>37.2</td><td><eq>27.0^a</eq></td><td>13.4</td></tr><tr><td>CLC</td><td>29.8</td><td>22.7</td><td>11.1</td><td>34.6</td><td><eq>31.8^a</eq></td><td>15.6</td></tr><tr><td>CNBSE</td><td>26.4</td><td>28.4</td><td>8.1</td><td>30.7</td><td>36.1</td><td>14.4</td></tr></table>


The lowest annotation rates of the AL strategies in each of the 3 datasets and the 2 low-difficulty entities are marked in bold. a RANDOM, CLUSTER, NBSE, and CLC cannot achieve the 99% target effectiveness for route in MADE 1.0 over all repetitions, we presented the values based on those iterations they can achieve. 


## Annotation rates comparisons based on target effectiveness across AL strategies by entity type

Next, the effectiveness of AL strategies for the low- and highdifficulty sub-groups of NEs was investigated. The entities in each sub-group are listed in the Tables 2 and 3. For the lowdifficulty entities, all AL strategies achieved a 99% target effectiveness for both entities on the 3 datasets under the experimental design, except for some runs in the route entity in the MADE 1.0 dataset. As shown in Table 2, the bestperforming strategy was CLUSTER, which achieved the lowest annotation rates for edits and entities. In general, LC and NBSE demonstrated low annotation rates for tokens. 


Table 3. Mean annotation rates for LC, NBSE, CLC, and CNSBE to achieve a 99% target effectiveness for high-difficulty entities across 3 datasets.


<table><tr><td rowspan="2">Annotation rate</td><td rowspan="2">AL strategy</td><td colspan="2">i2b2 2009</td><td colspan="2">n2c2 2018</td><td colspan="2">MADE 1.0</td></tr><tr><td>Reason (%)</td><td>Duration (%)</td><td>ADE (%)</td><td>Reason (%)</td><td>ADE (%)</td><td>Indication (%)</td></tr><tr><td rowspan="4">Edits</td><td>LC</td><td>22.1</td><td>17.7</td><td>14.7</td><td>23.9</td><td>7.6</td><td>10.3</td></tr><tr><td>NBSE</td><td>19.4</td><td>20.7</td><td>12.8</td><td>21.4</td><td>5.6</td><td>6.6</td></tr><tr><td>CLC</td><td>18.0</td><td>12.8</td><td>11.6</td><td>21.9</td><td>6.1</td><td>7.4</td></tr><tr><td>CNBSE</td><td>12.5</td><td>10.1</td><td>11.0</td><td>21.5</td><td>3.4</td><td>6.9</td></tr><tr><td rowspan="4">Tokens</td><td>LC</td><td>28.7</td><td>18.9</td><td>10.7</td><td>24.7</td><td>6.3</td><td>9.3</td></tr><tr><td>NBSE</td><td>22.1</td><td>22.1</td><td>10.0</td><td>21.7</td><td>5.6</td><td>6.6</td></tr><tr><td>CLC</td><td>26.8</td><td>15.2</td><td>10.7</td><td>26.1</td><td>7.7</td><td>9.3</td></tr><tr><td>CNBSE</td><td>18.1</td><td>13.5</td><td>11.3</td><td>27.0</td><td>5.3</td><td>9.3</td></tr><tr><td rowspan="4">Entities</td><td>LC</td><td>82.6</td><td>57.2</td><td>40.3</td><td>86.2</td><td>13.5</td><td>18.0</td></tr><tr><td>NBSE</td><td>67.1</td><td>67.7</td><td>38.7</td><td>76.3</td><td>12.2</td><td>13.8</td></tr><tr><td>CLC</td><td>76.5</td><td>42.6</td><td>37.1</td><td>87.2</td><td>14.0</td><td>16.5</td></tr><tr><td>CNBSE</td><td>52.0</td><td>37.4</td><td>39.3</td><td>90.1</td><td>8.8</td><td>16.2</td></tr></table>


The lowest annotation rates of the AL strategies in each of the 3 datasets and the 2 high-difficulty entities are marked in bold. 


For the high-difficulty entities, RANDOM and CLUSTER failed to reach 99% target effectiveness (note: these barely reached 93% for i2b2 2009 and n2c2 2018), which indicated their insufficiency in identifying informative samples for learning to differentiate more challenging entities. Uncertainty-based strategies and dynamic strategies can achieve a 99% target effectiveness or higher, as shown in Table 3. Considering the benefits of dynamic strategies, CNBSE reduced the annotation rate for edits by 22.5% on average compared to NBSE and CLC reduced the rate by 20.6% compared to LC. The learning curves for some selected entities are presented in Figures S7-S12. 

## Discussion

## AL strategies

The choice of AL strategies depends on the target performance and the ability to allocate effort to annotation. CLUS-TER requires the fewest edits to be made when a lower target effectiveness (98%) is anticipated. We chose Sentence-BERT to embed the sentences and performed cluster analysis in CLUSTER mainly due to its improved performance in representing the sentence semantically,34 which might result in diverse clusters during AL selection. However, CLUSTER could not achieve a high target performance overall, especially for challenging NEs such as reason, or ADE. During the selection process, CLUSTER may miss the chance to learn from high-difficulty NEs because the strategy is not driven by the model’s highest uncertainty. In contrast, uncertaintydriven strategies tend to select sentences containing NEs that the current NER model finds most challenging, and therefore these strategies can reach a high-performance target. For instance, considering ADE in the n2c2 2018 dataset, “leukopenia” represents a disease condition that may not be directly caused by drug intake, necessitating more contextual inference to determine if it’s an ADE. By selecting sentences with such entities, uncertainty-based AL strategies (LC, and NBSE) can achieve high target effectiveness. However, the uncertainty-driven nature will also result in a higher number of entity reviews and pre-annotation edits, particularly during the initial AL iterations when the NER model is still weak (as shown in Figures S5 and S6). 

The newly proposed dynamic AL strategies, CLC and CNBSE, which leverage the abovementioned scenarios, allow switching between AL algorithms, that is, from CLUSTER to 

LC or NBSE. This allows the selection of diverse samples until saturation, shifting focus to uncertain samples. The switch is determined by slowing down the decrease in the NER model’s training loss, indicating reduced improvement with CLUSTER. With this pool of diverse, informative samples, shifting the strategy to uncertainty-based strategies allows for identifying more uncertain informative samples. Therefore, dynamic strategies can distinguish challenging entities (an achievement not possible with a purely diversitybased strategy) while reducing annotation costs compared to traditional uncertainty-based strategies. The principle of shifting strategies holds when we use CLUSTER with LC or NBSE. By incorporating a strong uncertainty-based strategy NBSE, CNBSE in general performed the best in the highperformance target across all datasets. 

## AL and machine pre-annotations

While zero-shot or few-shot GPT models have demonstrated promising potential in clinical NER tasks, fine-tuning clinically pre-trained BERT-based models still demonstrates better performance trained on gold standard data.6 Furthermore, high-quality domain-specific annotation remains indispensable in the initial phase of large language models (LLMs) training. With AL and machine preannotations, less annotation effort is needed to create an accurate and reliable NER model when fine-tuning the pretrained language models, making AL applications more practical and invaluable in healthcare NLP applications. Further experiments can be conducted to evaluate whether our newly proposed AL strategies exhibit the same patterns in other pre-trained language models, for example, RoBERTa39 or XLNet,40 and on other clinical domain-specific datasets. At present, it is still uncertain how LLMs can be iteratively finetuned contributing to gold standard labeling for NER due to the closed-source nature of many LLMs such as GPT3.5 and GPT 4, high computation cost, and extensive training data requirements. However, the idea of selective sampling in AL may be integrated with few-shot LLMs and is worth further investigation in clinical NER. 

## Limitations and future works

This study presents some limitations. First, the study design cannot directly reveal the relationship between annotation time and annotation costs. Hence, future work will conduct a user study to quantify the time spent by different annotators for various tasks with varying annotation cost metrics. Some open-source, machine-assisted data labeling platforms, for example, Label Studio,41 could be used for the user study. Second, we adopted the standard settings of the BioClinical-BERT model without heavily tuning the hyperparameters to ensure training-time efficiency during annotation. Future studies may explore other options for leveraging the strength between various uncertainty-based AL strategies (eg, modified $\mathrm { L C } ^ { 2 3 } )$ and diversity-based AL strategies (eg, semantic similarity 20 

This study did not thoroughly investigate the threshold of training-loss decreases to determine the strategy switch in dynamic strategies. Future research will focus on tuning these hyperparameters. Furthermore, although the choice of the number of clusters in the CLUSTER strategy was examined (see Table S5), it is worth exploring a more objective way to select the number of clusters in the AL framework in the future. 



18. Shelmanov A, Liventsev V, Kireev D, et al. Active learning with deep pre-trained models for sequence tagging of clinical and biomedical texts. 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), San Diego, CA, USA. IEEE; 2019:482-489. 



## Conclusion



19. Kholghi M, Sitbon L, Zuccon G, Nguyen A. Active learning reduces annotation time for clinical concept extraction. Int J Med Inform. 2017;106:25-31. 



This study proposed a diversity-based strategy CLUSTER based on Sentence-BERT and 2 dynamic strategies (CLC, CNBSE) and investigated their effectiveness with the BioClinicalBERT model using 3 clinical datasets while considering different annotation costs. We demonstrated that when moderate effectiveness is desired, the newly proposed diversitybased strategy, CLUSTER, requires the fewest annotation edits. When a high target effectiveness is desired, the newly proposed dynamic AL strategies require fewer edits. 



20. Chen Y, Lasko TA, Mei Q, Denny JC, Xu H. A study of active learning methods for named entity recognition in clinical text. J Biomed Inform. 2015;58:11-18. 



## Acknowledgments



21. Culotta A, McCallum A. Reducing labeling effort for structured prediction tasks. Proceedings of the 20th National Conference on Artificial Intelligence, Pittsburgh, PA, USA. AAAI; 2005:746-751. 



Both authors express their gratitude to Neil Waters for editing a draft of this manuscript. 



22. Shen Y, Yun H, Lipton Z, Kronrod Y, Anandkumar A. Deep active learning for named entity recognition. Proceedings of the 2nd Workshop on Representation Learning for NLP, Vancouver, Canada. Association for Computational Linguistics; 2017:252-256. 



## Author contributions



23. Agrawal A, Tripathi S, Vardhan M. Active learning approach using a modified least confidence sampling strategy for named entity recognition. Prog Artif Intell. 2021;10(2):113-128. 



Jiaxing Liu and Zoie S.Y. Wong conceived and designed the research. Jiaxing Liu preprocessed the data, performed the simulation experiment, and evaluated the performance. Jiaxing Liu and Zoie S.Y. Wong analyzed and interpreted the results. Jiaxing Liu and Zoie S.Y. Wong wrote and revised the manuscript. 



24. Settles B, Craven MW. An analysis of active learning strategies for sequence labeling tasks. Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing, Honolulu, Hawaii. Association for Computational Linguistics; 2008:1070- 1079. 



## Supplementary material



25. Kholghi M, De Vine L, Sitbon L, Zuccon G, Nguyen A. Clinical information extraction using small data: an active learning approach based on sequence representations and word embeddings. J Assoc Inf Sci Technol. 2017;68(11):2543-2556. 



Supplementary material is available at Journal of the American Medical Informatics Association online. 



26. Chen Y, Lask TA, Mei Q, et al. An active learning-enabled annotation system for clinical named entity recognition. BMC Med Inform Decis Mak. 2017;17(Suppl 2):82. 



## Funding



27. Wei Q, Chen Y, Salimi M, et al. Cost-aware active learning for named entity recognition in clinical text. J Am Med Inform Assoc. 2019;26(11):1314-1322. 



This work was supported by the Natural Science Foundation of Hubei Province (Grant No. 2023AFB414), “the Fundamental Research Funds for the Central Universities” in the Zhongnan University of Economics and Law (Grant No. 2722023BQ053), and the Japan Society for the Promotion of Science KAKENHI (Grant No. 18H03336). 



28. Shen D, Zhang J, Su J, Zhou G, Tan CL. Multi-criteria-based active learning for named entity recognition. Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), Barcelona, Spain. Association for Computaional Linguistics; 2004:589-596. 



## Conflicts of interest



29. Uzuner O, Solti I, Cadag E. Extracting medication information from clinical text. J Am Med Inform Assoc. 2010;17(5):514-518. 



None declared. 

## Data availability



30. Jagannatha A, Liu F, Liu W, Yu H. Overview of the first natural language processing challenge for extracting medication, indication, and adverse drug events from electronic health record notes (MADE 1.0). Drug Saf. 2019;42(1):99-111. 



Our code is available upon request. The public datasets used in this study are available upon request to their providers, following the respective data policies. 



31. Devlin J, Chang M-W, Lee K, Toutanova K. BERT: pre-training of deep bidirectional transformers for language understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, Minneapolis, MN, USA. Association for Computational Linguistics; 2019:4171-4186. 



## References



32. Ren P, Xiao Y, Chang X, et al. A survey of deep active learning. ACM Comput Surv. 2021;54(9):1-40. 





33. Kim S, Song Y, Kim K, Cha J-W, Lee GG. MMR-based active machine learning for bio named entity recognition. Proceedings of the Human Language Technology Conference of the NAACL, Companion Volume: Short Papers, New York City, USA. Association for Computational Linguistics; 2006:69-72. 





1. Henry S, Buchan K, Filannino M, Stubbs A, Uzuner O. 2018 n2c2 shared task on adverse drug events and medication extraction in electronic health records. J Am Med Inform Assoc. 2019;27 (1):3-12. 





34. Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using Siamese BERT-networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China. Association for Computational Linguistics; 2019:3982-3992. 





2. Wu S, Roberts K, Datta S, et al. Deep learning in clinical natural language processing: a methodical review. J Am Med Inform Assoc. 2019;27(3):457-470. 





35. Yu G, Yang Y, Wang X, et al. Adversarial active learning for the identification of medical concepts and annotation inconsistency. J Biomed Inform. 2020;108:103481. 





03. Gobbel GT, Garvin J, Reeves R, et al. Assisted annotation of medical free text using RapTAT. J Am Med Inform Assoc. 2014;21 (5):833-841. 





36. Levenshtein VI. Binary codes capable of correcting deletions, insertions, and reversals. Sov Phys Dokl. 1966;10(8):707-710. 





4. Lingren T, Deleger L, Molnar K, et al. Evaluating the impact of pre-annotation on annotation speed and potential bias: natural language processing gold standard development for clinical named entity recognition in clinical trial announcements. J Am Med Inform Assoc. 2014;21(3):406-413. 





37. Kholghi M, Sitbon L, Zuccon G, Nguyen A. Active learning: a step towards automating medical concept extraction. J Am Med Inform Assoc. 2015;23(2):289-296. 





05. Gilardi F, Alizadeh M, Kubli M. ChatGPT outperforms crowd workers for text-annotation tasks. Proc Natl Acad Sci. 2023;120 (30):e2305016120. 





38. seqeval: a Python framework for sequence labeling evaluation; 2018. Accessed July 20, 2024. https://github.com/chakki-works/ seqeval 





6. Hu Y, Chen Q, Du J, et al. Improving large language models for clinical named entity recognition via prompt engineering. J Am Med Inform Assoc. 2024:ocad259. 





39. Liu Y, Ott M, Goyal N, et al. RoBERTa: a robustly optimized Bert pretraining approach. arXiv, arXiv:1907.11692, 2019, preprint: not peer reviewed. 





07. Chen Q, Sun H, Liu H, et al. An extensive benchmark study on biomedical text generation and mining with ChatGPT. Bioinformatics. 2023;39(9):btad557. 





40. Yang Z, Dai Z, Yang Y, Carbonell J, Salakhutdinov RR, Le QV. XLNet: generalized autoregressive pretraining for language understanding. Proceedings of the 33rd International Conference on Neural Information Processing Systems, Vancouver, Canada. Red Hook, NY: Curran Associates Inc.; 2019:5753-5763. 





8. Alsentzer E, Murphy J, Boag W, et al. Publicly available clinical BERT embeddings. Proceedings of the 2nd Clinical Natural Language Processing Workshop, Minneapolis, MN, USA. Association for Computational Linguistics; 2019:72-78. 





41. Label Studio. Open Source Data Labeling Platform; 2024. Accessed July 20, 2024. https://labelstud.io/ 





09. Gal Y, Islam R, Ghahramani Z. Deep Bayesian active learning with image data. Proceedings of the 34th International Conference on Machine Learning, Sydney, NSW, Australia. PMLR; 2017:1183-1192. 





10. Beluch WH, Genewein T, Nurnberger A, Kohler JM. The power of ensembles for active learning in image classification. 2018 IEEE/ CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA. IEEE; 2018:9368-9377. 





11. Figueroa RL, Zeng-Treitler Q, Ngo LH, Goryachev S, Wiechmann EP. Active learning for clinical text classification: is it better than random sampling? J Am Med Inform Assoc. 2012;19(5):809-816. 





12. Nguyen DHM, Patrick JD. Supervised machine learning and active learning in classification of radiology reports. J Am Med Inform Assoc. 2014;21(5):893-901. 





13. Ein-Dor L, Halfon A, Gera A, et al. Active learning for BERT: an empirical study. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), online. Association for Computational Linguistics; 2020:7949-7962. 





14. Gu�elorget P, Grilheres B, Zaharia T. Deep active learning with simulated rationales for text classification. International Conference on Pattern Recognition and Artificial Intelligence. Cham: Springer; 2020:363-379. 





15. Weissenbacher D, Ge S, Klein A, et al. Active neural networks to detect mentions of changes to medication treatment in social media. J Am Med Inform Assoc. 2021;28(12):2551-2561. 





16. Liu Y, Hu J, Chen Z, Wan X, Chang T-H. EASAL: entity-aware subsequence-based active learning for named entity recognition. Proc AAAI Conf Artif Intell. 2023;37(7):8897-8905. 





17. Liu M, Tu Z, Zhang T, Su T, Xu X, Wang Z. LTP: a new active learning strategy for CRF-based named entity recognition. Neural Process Lett. 2022;54(3):2433-2454. 

