## 1 Introduction


## 2 Setup

### 2.1 Running locally

First, install the required packages
```
pip install -r requirements.txt
```

Then download the word2vec wikipedia-pubmed pretrained embeddings:
```
./download_embeddings.sh
```

Then, to pretrain on the pubmed20k dataset (see pubmed_text_classification/scripts/train_and_evaluate.py for a number of command-line arguments):
```
cd pubmed_text_classification/scripts
python train_and_evaluate.py
```

### 2.2 Running in google colab

First download the embeddings, and put them in your Google Drive in a folder called 'pretrained_embeddings'. Then the following code should work:
```python
# mount google drive
from google.colab import drive
drive.mount('/content/gdrive')

# clone the repo
```
```
!git clone --single-branch --branch refactor https://github.com/cjs220/pubmed_text_classification.git
```
```python
# pretrain on pubmed20k
import os
os.chdir('pubmed_text_classification/pubmed_text_classification/scripts')
```
```
!python '/content/gdrive/My Drive/pretrained_embeddings/wikipedia-pubmed-and-PMC-w2v.bin'
```
