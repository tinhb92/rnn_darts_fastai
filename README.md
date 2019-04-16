# DARTS for RNN with fastai

Language model on Penn Treebank using Differentiable Architecture Search (DARTS) and fastai library.\
[Blog post](https://medium.com/p/differentiable-architecture-search-for-rnn-with-fastai-a5e247aeb937?source=email-69f5f36d8a7c--writer.postDistributed&sk=fb4a8d77b9ffd5bf82a65747cc4aec76).\
Based on [DARTS: Differentiable Architecture Search ](https://openreview.net/pdf?id=S1eYHoC5FX) by Hanxiao Liu, Karen Simonyan, Yiming Yang.\
Check out the [original implementation](https://github.com/quark0/darts).

## Requirements
fastai 1.0.52.dev0 (latest as of 10th April 2019), PyTorch 1.0.

## Instructions 
1. Run databunch_nb.ipynb to create databunch
2. Run train_search_nb.ipynb to search for genotype. ~5 hours on 1 v100 gpu for 1 run.\
RNN search is sensitive to initialization so there should be several runs with different seed
3. Train that genotype from scratch on train_nb.ipynb. ~1.5 days for 1600 epochs.
4. Test a model using test_nb.ipynb

## Pretrained model
Pretrained model of DARTS_V1 genotype after 600 epochs
[darts_V1.pth](https://drive.google.com/file/d/1hOs932fEbENlm3mzFTlOzAHtaPxjP690/view?usp=sharing).\
Place the file at data/models and run test_nb.ipynb. Loss ~4.22, 68.0 perplexity.\
Caveat: I haven't been able to get ~58.0 test perplexity like the original implementation.

## fastai dev version installation
```bash
git clone https://github.com/fastai/fastai
cd fastai
tools/run-after-git-clone
pip install -e ".[dev]"
git pull 
```
