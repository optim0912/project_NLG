#!/bin/bash

# This is a shell script that preprocesses the dataset. 
# This creates a Korean corpus and an English corpus to be used for train, valid, and test. 

echo "[Text Concatenation]"
cat ./dataset_process/original_dataset/*.txt > ./dataset_process/corpus.tsv
echo "Complete"
echo

echo "[Shuffle]"
shuf ./dataset_process/corpus.tsv > ./dataset_process/corpus_shuf.tsv
echo "Complete"
echo

echo "[Split into train, valid and test]"
head -n 1200000 ./dataset_process/corpus_shuf.tsv > ./dataset_process/corpus_shuf_train.tsv
tail -n 402409 ./dataset_process/corpus_shuf.tsv | head -n 200000 > ./dataset_process/corpus_shuf_valid.tsv
tail -n 202409 ./dataset_process/corpus_shuf.tsv > ./dataset_process/corpus_shuf_test.tsv
echo "Complete"
echo

echo "[Split into English Corpus and Korean Corpus]"
cut -f1 ./dataset_process/corpus_shuf_train.tsv > ./dataset_process/corpus_shuf_train_ko
cut -f2 ./dataset_process/corpus_shuf_train.tsv > ./dataset_process/corpus_shuf_train_en
cut -f1 ./dataset_process/corpus_shuf_valid.tsv > ./dataset_process/corpus_shuf_valid_ko
cut -f2 ./dataset_process/corpus_shuf_valid.tsv > ./dataset_process/corpus_shuf_valid_en
cut -f1 ./dataset_process/corpus_shuf_test.tsv > ./dataset_process/corpus_shuf_test_ko
cut -f2 ./dataset_process/corpus_shuf_test.tsv > ./dataset_process/corpus_shuf_test_en
echo "Complete"
echo

echo "[Tokenization]"
# Apply post processing for detokenization after tokenization.

# Tokenization for English Corpus
cat ./dataset_process/corpus_shuf_train_en | python ./dataset_process/tokenizer.py | python ./dataset_process/post_tokenize.py ./dataset_process/corpus_shuf_train_en > ./dataset_process/corpus_shuf_train_tok_en &
cat ./dataset_process/corpus_shuf_valid_en | python ./dataset_process/tokenizer.py | python ./dataset_process/post_tokenize.py ./dataset_process/corpus_shuf_valid_en > ./dataset_process/corpus_shuf_valid_tok_en &
cat ./dataset_process/corpus_shuf_test_en | python ./dataset_process/tokenizer.py | python ./dataset_process/post_tokenize.py ./dataset_process/corpus_shuf_test_en > ./dataset_process/corpus_shuf_test_tok_en &

# Tokenization for Korean Corpus
cat ./dataset_process/corpus_shuf_train_ko | mecab -O wakati -b 99999 | python ./dataset_process/post_tokenize.py ./dataset_process/corpus_shuf_train_ko > ./dataset_process/corpus_shuf_train_tok_ko &
cat ./dataset_process/corpus_shuf_valid_ko | mecab -O wakati -b 99999 | python ./dataset_process/post_tokenize.py ./dataset_process/corpus_shuf_valid_ko > ./dataset_process/corpus_shuf_valid_tok_ko &
cat ./dataset_process/corpus_shuf_test_ko | mecab -O wakati -b 99999 | python ./dataset_process/post_tokenize.py ./dataset_process/corpus_shuf_test_ko > ./dataset_process/corpus_shuf_test_tok_ko &
wait
echo "Complete"
echo

echo "[Learn BPE model]"
python ./dataset_process/learn_bpe.py --input ./dataset_process/corpus_shuf_train_tok_en --output ./dataset_process/bpe_model_en --symbols 50000 &
python ./dataset_process/learn_bpe.py --input ./dataset_process/corpus_shuf_train_tok_ko --output ./dataset_process/bpe_model_ko --symbols 30000 &
wait
echo "Complete"
echo

echo "[Apply BPE]"
# Subword Segmentation for English Corpus
cat ./dataset_process/corpus_shuf_train_tok_en | python ./dataset_process/apply_bpe.py -c ./dataset_process/bpe_model_en > ./dataset_process/corpus_shuf_train_tok_bpe_en &
cat ./dataset_process/corpus_shuf_valid_tok_en | python ./dataset_process/apply_bpe.py -c ./dataset_process/bpe_model_en > ./dataset_process/corpus_shuf_valid_tok_bpe_en &
cat ./dataset_process/corpus_shuf_test_tok_en | python ./dataset_process/apply_bpe.py -c ./dataset_process/bpe_model_en > ./dataset_process/corpus_shuf_test_tok_bpe_en &

# Subword Segmentation for Korean Corpus
cat ./dataset_process/corpus_shuf_train_tok_ko | python ./dataset_process/apply_bpe.py -c ./dataset_process/bpe_model_ko > ./dataset_process/corpus_shuf_train_tok_bpe_ko &
cat ./dataset_process/corpus_shuf_valid_tok_ko | python ./dataset_process/apply_bpe.py -c ./dataset_process/bpe_model_ko > ./dataset_process/corpus_shuf_valid_tok_bpe_ko &
cat ./dataset_process/corpus_shuf_test_tok_ko | python ./dataset_process/apply_bpe.py -c ./dataset_process/bpe_model_ko > ./dataset_process/corpus_shuf_test_tok_bpe_ko &
wait
echo "Complete"
echo

echo "[Check the Final Corpus]"
head -n 2 ./dataset_process/corpus_shuf_train_tok_bpe_*
head -n 2 ./dataset_process/corpus_shuf_valid_tok_bpe_*
head -n 2 ./dataset_process/corpus_shuf_test_tok_bpe_*
wc -l ./dataset_process/corpus_shuf_*_tok_bpe_*
echo "Complete"
echo

echo "[Split Train Corpus]"
split -l 300000 -d ./dataset_process/corpus_shuf_train_tok_bpe_ko ./dataset_process/corpus_shuf_train_tok_bpe_ko_
split -l 300000 -d ./dataset_process/corpus_shuf_train_tok_bpe_en ./dataset_process/corpus_shuf_train_tok_bpe_en_
mv ./dataset_process/corpus_shuf_train_tok_bpe_*_* ./dataset
mv ./dataset_process/corpus_shuf_valid_tok_bpe_* ./dataset
mv ./dataset_process/corpus_shuf_test_tok_bpe_* ./dataset
echo "Complete"
echo
