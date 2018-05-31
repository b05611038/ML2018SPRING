wget 'https://www.dropbox.com/s/z2fm02ocx9py8it/rnn_model.h5?dl=0' -O 'rnn_model.h5'
wget 'https://www.dropbox.com/s/olrkqnkva8r4ilz/word2vec_model.bin?dl=0' -O 'word2vec_model.bin'
python3 hw5_test.py $1 $2 $3
