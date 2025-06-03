# Gradient-guided Unsupervised Text Style Transfer via Contrastive Learning

[paper link](https://arxiv.org/pdf/2202.00469)

Since the paper wasn’t accepted and I was quite busy that year, I ended up abandoning this work. Also, this was one of my earlier projects, so the code isn’t very well organized — I apologize for that.

## How to use

- To do style transfer for **yelp** dataset

  cd `method/yelp` and run `python main.py --attack --not_nll --pos_victim_lr=0.2 --neg_victim_lr=0.2`

- The outputs are stored in the **outputs** folder, the outputs sentences (\*\_out) are automatically exrtacted from the log files (\*\_log). To reimplement the extraction process, run `python get_i_ter.py --file=FILENAME --pos_c=POS_C --neg_c=NEG_C`, where:

  - **FILENAME** is the log file to extract from
  - **POS_C** and **NEG_C** are two hyperparameters, specifying different values results in different output file

  For example, cd `outputs` and run `python get_i_iter.py --file=yelp_log --pos_c=2 --neg_c=2` 

## How to cite
```bash
@misc{fan2022gradientguidedunsupervisedtextstyle,
      title={Gradient-guided Unsupervised Text Style Transfer via Contrastive Learning}, 
      author={Chenghao Fan and Ziao Li and Wei wei},
      year={2022},
      eprint={2202.00469},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2202.00469}, 
}
```
