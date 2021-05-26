# Codenames

## Files
* *codenames.json:* three annotated games of codenames
* *semantic_models.py:* includes classes for Word2Vec and GloVe embeddings
* *baseline_model.py:* codemaster and guesser model adapted from
    - Kim, A., Ruzmaykin, M., Truong, A., & Summerville, A. (2019, October). Cooperation and codenames: Understanding natural language processing via codenames. In Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (Vol. 15, No. 1, pp. 160-166).

## How to run baseline
* Download the GloVe dataset here: http://nlp.stanford.edu/data/glove.42B.300d.zip
* Unzip and put it in the codenames directory
* Get file with 10,000 most common English words here: https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt and put it in the codenames directory
* Install python packages:
    - numpy
    - scipy
    - gensim
* run *baseline_model.py*