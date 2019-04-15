# darts_rnn_fastai

Language model on PTB using DARTS and fastai library

rnn search is sensitive to init
 
run databunch.ipynb to create databunch
run train_search_nb.ipynb to search for genotype, around 4.5 hours on gcp instances v100 gpu to finish 1
the author recommend 4 runs with different seed
train that genotype from scratch on train_nb.ipynb
test a model using test_nb.ipynb
 
fastai 1.0.52.dev0 (latest as of 10/4/2019), pytorch 1.0

fastai installation dev version

```bash
git clone https://github.com/fastai/fastai
cd fastai
tools/run-after-git-clone
pip install -e ".[dev]"
git pull 
```


