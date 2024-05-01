# rsFC_in_internetaddiction

Masters Thesis Assessing Neural Biomarkers for Internet Addiction via Predictive Modeling

In the project a published dataset by Mendes et al. (N = 186) was used to predict Internet Addiction (by IAT; Young, 1998) out of resting state functional connectivity (fMRI) by Gradient Boosted Decision Trees. The goal was to find neural biomarkers of this disease and to test the role of the reward-, self-reference and attention-network.

## Abstract
Due to increasing omnipresence of digital services, numbers of people frequently using the Internet grow steadily and so does the prevalence of Internet Addiction (IA), a non-substance
dependency. While the Internet may be a supportive tool for many, others suffer from excessive and addictive use accompanied by depression, anxiety and loneliness, however to date the underlying
neural mechanisms are not fully understood. This work aims to fill the gap by applying a gradient boosted decision tree machine learning regressor trying to predict IA out of resting-state functional
connectivity (rsFC). Successful prediction would contribute to the framework of digital phenotyping helping to identify relevant networks and robust markers of psychologic dysfunctionality, more valid
than self-report questionnaires. Therefore, a previously collected sample of N=186 participants, each providing 3*15min resting state scans was analyzed hypothesizing that three networks are relevant
for predicting IA, namely the reward-network, self-referential-network and attention-network. In a nested cross-validation scheme, both an original and a null model were fitted. The original model
showed significantly better performance than the shuffled model (MAE: M = 6.76 ± 0.697 vs. M = 7.497 ± 0.708, p < .001; R2: M = .086 ± .104 vs. M = -.105 ± .103, p < .001) however original feature
importance for the respective networks did not significantly exceed that of the shuffled model, leading to the rejection of all hypotheses. As prediction mainly failed due to the distribution of the
outcome scale, this work underlines the need for data examination in modeling tasks. 

**Words**: 248

**Keywords**: Predictive Modeling, Functional Connectomes, Internet Addiction, Social-Media, rsfMRI

## Codefiles
The codefiles are:

**functions.py** - Functions to retrieve, preprocess and analyse data which is called by the other scripts to keep them clean. Authored by Luis Glenzer

**analysis** - Script for statistical analysis of the data to answer research question and test hypotheses. Authored by Luis Glenzer

**ml_analysis.py** - Code for machine learning algorithm incorporating k-fold CV (extra script). Authored by David Steyrl 

**sklearn_repeated_group_k_fold.py**  - Code for *k*-fold Cross-Validation. Authored by David Steyrl

Plots relevant for the work are saved in the **plots** folder
