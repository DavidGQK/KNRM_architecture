# Implementation of KNRM architecture

The dataset `Quora Question Pairs (QQP)` is part of the `GLUE` dataset used for comprehensive evaluation of text-related machine learning models.
The `GloVe 6B` vectors will be used as text embeddings  <br/>
There are three levels of relevance:
- `2` - the question is a complete duplicate (according to the original markup, these are pairs with a target equal to `1`)
- `1` - the question is very similar to the original question, but is not a complete duplicate (according to the original markup, these are pairs with a target of `0`)
- `0` - the question is not similar to the original question, irrelevant (there are no such pairs in the dataset, you can generate them yourself from the general corpus of all questions)
## Model training
A train method that iterates `N` iterations over the training `Dataloader`. The sample is recreated every `change_train_loader_ep` epoch. The training sample itself is created in the `sample_data_for_train_iter` method. `Triplets of documents` are picked up in `PairWise` mode. During training at the end of an epoch, the valid method is used to calculate the `NDCG`. Model training ends when `NDCG` is greater than 0.93 <br/>
## Implementation details and description
`min_token_occurances` - is the minimum number of times a word (token) must appear in the sample in order not to be discarded as low-frequency. If the value is equal to one, all words which are represented in the dataset will stay <br/>
`emb_rand_uni_bound` - half of interval width, from which embedding vectors are uniformly generated (if the vector is not represented in the `GloVe` set). If the parameter is `0.2`, then each component of the vector belongs to `U(-0.2,0.2)` <br/>
`freeze_knrm_embeddings` - flag indicating whether embeddings should be retrained, whether gradients will be counted by them (if `True` there will be no retraining) <br/>
`knrm_kernel_num `- the number of cores in `KNRM` <br/>
`knrm_out_mlp` - configuration of the `MLP-layer` in the `KNRM` output <br/>
`dataloader_bs` - the size of the batches when training and validating the model <br/>
`train_lr` - `learning Rate`, used when training the `KNRM` model <br/>
`change_train_loader_ep` - how often to change/regenerate a sample for model training <br/>
`handle_punctuation` - clears the string from punctuation <br/>
`simple_preproc` - full line preprocessing <br/>
`get_all_tokens` - method that generates a list of ALL tokens represented in the datasets (in `pd.DataFrame`) fed in. To implement it, form a unique set of all texts, then calculate the frequency of each token (i.e. after processing simple_preproc) and cut off those that do not pass the threshold equal to `min_token_occurancies`, using the `_filter_rare_words` method. The output is a list of tokens for which embeddings will be generated and into which the original question texts will be split <br/>
`_read_glove_embeddings` - read embeddings file into a dictionary, where the key is a word and the value is an embedding vector <br/>
`create_glove_emb_from_file` - the method generates matrix (matrix of embeddings of size `N∗D`, where `N` is number of tokens, `D` is embedding dimension), vocab (dictionary of size `N`, matching each word with embedding index), `unk_words` is a list of words that were not in the original embeddings, and therefore had to generate a random embedding for them from a uniform distribution (or another vector with specified characteristics, see below) <br/>
Two special tokens, `PAD` and `OOV`, with indices `0` and `1`, respectively, were added to the dictionary. The first token is used to fill voids in tensors (when one question consists of more tokens than the second, but they must be represented as a matrix, where the rows have the same length) and must consist entirely of zeros. The second token is used for the tokens that are not in the dictionary <br/>
`GaussianKernel` - does not contain any trainable parameters and is a simple non-linear operator <br/>
`_get_kernels_layers` - generates the list of all kernels (`K ones`), used in the algorithm <br/>
`_get_mlp` - generates an output `MLP-layer` for ranking based on the result of Kernels <br/>
`_get_matching_matrix` - generates an every-other-matter interaction matrix between words of one and two questions (query and document). It uses cosine similarity between embeddings of individual tokens as a measure <br/>
`_apply_kernels` - applies kernels to `matching_matrix` according to the formula in theory <br/>

`Dataset` and `Dataloader` are important parts of `PyTorch` learning pipelines. They are smart and flexible wrappers over data that allow you to iterate over a dataset. The first works with document triplets (`source question`, `question-candidate 1` and `question-candidate 2` for `PairWise-mode learning`), the second works with pairs (evaluates separately the relevance of the question-candidate to the source question) <br/>

`idx_to_text_mapping` - responsible for matching the index (`id_left` and `id_right`) with the text <br/>
`vocab` - word mapping into the index (namely the word indexes are fed to the embedding layer `KNRM` as inputs and the required matrix row is taken by them) <br/>
`oov_val` - the value (index) in the dictionary in case the word is not represented in the dictionary <br/>
`preproc_func` - text processing and tokenization function <br/> 
`max_len` - maximal number of tokens in the text <br/>
`__getitem__` - returns the set of tokens for a given pair or triplet of several id and token, where the tokens are expressed as indices of words in the dictionary <br/> 
`_convert_text_idx_to_token_idxs` - translates `id_left`/`id_right` into indexes of tokens in the dictionary, in particular with `_tokenized_text_to_index` (translating processed text after `preproc_func` into indexes) <br/>

`__getitem__` in the class heirs, `TrainTripletsDataset` and `ValPairsDataset`, differs in that document triples are used for training and pairs for validation. The pairs and triples themselves must be input to `index_pairs_or_triplets`. This is a list of lists of id and labels. The output of this method expects one or two dictionaries with the keys query and document, as well as a target label (for training - the answer to whether the first document is really more relevant to the query than the second, for validation - relevance from `0` `to` `2`)<br/>

`collate_fn` - assembles the batches from several training examples for `KNRM`. It takes as input a list of outputs from the datasets above and forms from them a single dict with tensors as values. This function should be passed, among other things, to the `DataLoader` in order to automatically collect, for example, 128 objects (triplets in training) into one dataset, which will be fed to `KNRM`
