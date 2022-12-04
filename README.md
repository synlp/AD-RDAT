# AD-RDAT
Implementation of paper "[Improving Arabic Diacritization with Regularized Decoding and Adversarial Training](https://aclanthology.org/2021.acl-short.68/)" at ACL-2021

## Citation

```
@inproceedings{qin-etal-2021-improving,
    title = "Improving Arabic Diacritization with Regularized Decoding and Adversarial Training",
    author = "Qin, Han and Chen, Guimin and Tian, Yuanhe and Song, Yan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    pages = "534--542",
}
```

## Requirements

Our code works with `python 3.8` and requires the following packages: [sklearn](https://scikit-learn.org/stable/install.html), [pytorch](https://pytorch.org/).

It also require the PyTorch version of pre-trained language models: [multi-lingual BERT](https://github.com/google-research/bert) and [AraBERT](https://github.com/aub-mind/arabert).

## Usage

See the commands in `run.sh` to train a model on the small sample data.
