import numpy as np
import pandas as pd
import string
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt

def read_text(filename):
    file = open(filename, mode = 'rt', encoding='utf-8')
    text = file.read()
    file.close()
    sents = text.strip().split('\n')
    return sents

data_hin = read_text('parallel-n/IITB.en-hi.hi')
data_eng = read_text('parallel-n/IITB.en-hi.en')

df_hin = pd.DataFrame(data_hin)
df_eng = pd.DataFrame(data_eng)
# print(df_eng)
# print(df_hin)

df_hin[0]=df_hin[0].str.replace('[{}]'.format(string.punctuation), '')
df_eng[0]=df_eng[0].str.replace('[{}]'.format(string.punctuation), '')

df_hin[0] = df_hin[0].str.lower()
df_eng[0] = df_eng[0].str.lower()

hin_zero=[]
eng_zero=[]

c=0
for i in df_eng[0].str.strip().astype(bool):
    if(i==False):
        eng_zero.append(c)
    c+=1
    pass
c=0
for i in df_hin[0].str.strip().astype(bool):
    if(i==False):
        hin_zero.append(c)
    c+=1
    pass

remove_list=list(set(eng_zero+hin_zero))
df_eng.drop(df_eng.index[remove_list], inplace=True)

df_eng.reset_index(inplace=True)
df_hin.drop(df_hin.index[remove_list], inplace=True)
df_hin.reset_index(inplace=True)

eng_tokenizer = Tokenizer(df_eng[0][:50000])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = 8
print('Size of English Vocabulary : %d' % eng_vocab_size)

hin_tokenizer = Tokenizer(df_hin[0][:50000])
hin_vocab_size = len(hin_tokenizer.word_index) + 1
hin_length = 8
print('Size of Hindi Vocabulary : %d' % hin_vocab_size)
