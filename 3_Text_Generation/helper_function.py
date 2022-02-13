# This is the helper function file for the text generation project

# --------------------------------------- GPT2 -----------------------------------------------
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split

# Function for extracting the text column and building text files (used in ```prepapre_data``` function)
def build_text_files(dataframe, dest_path, text_col):
    f = open(dest_path, 'w')
    data = ''
    for texts in dataframe[text_col]:
        data += texts + "  "
    f.write(data)
    
def prepare_data(df_name, seed=None, custom_symbol=None):
    # 1. Specify text column and other relevant columns
    text_col_dict = {'news_data':'text',
                     'netflix_data':'description',
                     'lyrics_data':'lyric'}
    text_col = text_col_dict[df_name]
    # relevant columns
    keep_col_list = {'netflix_data':['title','director'],
                     'news_data':['TITLE','URL'], 
                     'lyrics_data':['song_name','artist']}
    
    # 2. Load in dataset
    orig_data = pd.read_csv('./fine_tune_data/'+df_name+'.csv')
    orig_data = orig_data[keep_col_list[df_name]+[text_col]]
    print('We are now using "{}" with text column name "{}"'.format(df_name, text_col))
    # 2.1 Custom processing for news and lyrics data
    if df_name=='news_data':
        # keep only news with more than 10 words (otherwise probably not news)
        crit = orig_data[text_col].str.split().str.len()>10
        orig_data = orig_data[crit]
        # filter out the non-English news (if it contains non-english)
        # extra_symbol = ['“','”',"’","'",'£',"‘",'…','—','–','‐','€','¥','•','·']
        if custom_symbol:
            char_set = string.printable+string.whitespace+''.join(custom_symbol)
        else: char_set = string.printable+string.whitespace
        orig_data['hasSpeialChar'] = orig_data[text_col].apply(lambda x: 0 if all(char in char_set for char in x) else 1)
        orig_data = orig_data[orig_data.hasSpeialChar==0]
        # randomly sample df to keep runtime within restriction
        orig_data = orig_data[crit].sample(frac=0.8, random_state=seed)
    elif df_name=='lyrics_data':
        # keep only lyrics with more than 100 words and randomly sample df to keep runtime within restriction
        crit = orig_data[text_col].str.split().str.len()>100
        orig_data = orig_data[crit].sample(frac=0.3, random_state=seed)
        
    # 3. For all datasets, remove too long/short text (outlier: Interquartile Range Method with cut-off 1.5*iqr)
    length_stat = orig_data[text_col].str.split().str.len().describe()
    q1, q3 = length_stat['25%'], length_stat['75%']
    iqr = q3-q1
    crit_lower = orig_data[text_col].str.split().str.len()>=q1-1.5*iqr
    crit_upper = orig_data[text_col].str.split().str.len()<=q3+1.5*iqr
    orig_data = orig_data[crit_lower&crit_upper]

    # 4. Split the original dataset into train, val, and test (60%, 30%, 10%)
    train, test = train_test_split(orig_data, test_size=0.4, random_state=seed)
    val, test = train_test_split(test, test_size=0.25, random_state=seed)
    print("Dataset shape (train/val/test)': ", 
          '('+str(len(train))+', '+str(len(val))+', '+str(len(test))+')')
    # display num_word distribution in train set
    print("\nDistribution of n_word in the training set")
    print(train[text_col].str.split().str.len().describe())
    
    # 5. Extract the text column and build text files
    txt_file_path = './built_txt_file/'+df_name
    build_text_files(train, txt_file_path+'_train_dataset.txt', text_col=text_col)
    build_text_files(val, txt_file_path+'_val_dataset.txt', text_col=text_col)
    
    # 6. View some text data
    print("\n"+"Example Text"+"\n"+"-"*110)
    print(train[text_col].sample(random_state=123).values[0]+"\n"+"-"*110)

    return text_col, train, val, test

from transformers import TextDataset,DataCollatorForLanguageModeling

def load_dataset(train_path,val_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    val_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=val_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,val_dataset,data_collator

# --------------------------------------------------------------------------------------------

# ---------------------------------------- RNN -----------------------------------------------
import tensorflow as tf
import os
import time

# Model building
class custom_rnn(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
        
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def build_rnn(text, ids_from_chars):
    # 1. prepare for embedding
    vocab = sorted(set(text))
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    
    # 2. model hyperparameters
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    
    # 3. build the rnn model
    model = custom_rnn(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    
    return dataset, model


# Generation
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
    @tf.function
    def generate_one_step(self, inputs, seed, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask
        
        # Sample the output logits to generate token IDs.
        tf.random.set_seed(seed)
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1, seed=seed)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states
    
# --------------------------------------------------------------------------------------------

# ------------------------------- Result Visulization ------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
    
def plot_results_bleu(data1, data2, data3, save_name='BLEU score for all datasets'):
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Adjust the subplot layout parameters
    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    data_melt1 = data1[['gpt2_BLEU','rnn_BLEU']].reset_index()
    data_melt1 = data_melt1.melt(id_vars='index', var_name='model', value_name='BLEU')
    sns.set_palette("colorblind")
    g1 = sns.lineplot(x='index',y='BLEU', hue = 'model', data=data_melt1, linewidth=0.8, ax=axs[0])
    g1.set_yticks(ticks=(3*10**(-231), 3*10**(-232)))
    g1.set(title="new_result")
    
    data_melt2 = data2[['gpt2_BLEU','rnn_BLEU']].reset_index()
    data_melt2 = data_melt2.melt(id_vars='index', var_name='model', value_name='BLEU')
    g2 = sns.lineplot(x='index',y='BLEU', hue = 'model', data=data_melt2, linewidth=0.8, ax=axs[1])
    g2.set_yticks(ticks=(3*10**(-231), 3*10**(-232)))
    g2.set(title="netflix_result")
    
    data_melt3 = data3[['gpt2_BLEU','rnn_BLEU']].reset_index()
    data_melt3 = data_melt3.melt(id_vars='index', var_name='model', value_name='BLEU')
    # sns.set_palette("colorblind")
    g3 = sns.lineplot(x='index',y='BLEU', hue = 'model', data=data_melt3, linewidth=0.8, ax=axs[2])
    g3.set_yticks(ticks=(3*10**(-231), 3*10**(-232)))
    g3.set(title="lyrics_result")

    fig.suptitle('Graph 1. '+save_name)
    plt.savefig('./text_generation_results/'+'_'.join(save_name.split()))

def plot_results_gmerr(data1, data2, data3, save_name='Grammar Error Rate for all datasets'):
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Adjust the subplot layout parameters
    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    g1 = sns.boxplot(fliersize=2, width=0.5, showmeans=True, meanprops={"marker":"s","markerfacecolor":"white"},
                     data=data1[['gpt2_grammar_error','rnn_grammar_error']], ax=axs[0])
    g1.set_xticklabels(g1.get_xticklabels())
    g1.set(title="news_result")
    
    g2 = sns.boxplot(fliersize=2, width=0.5, showmeans=True, meanprops={"marker":"s","markerfacecolor":"white"},
                data=data2[['gpt2_grammar_error','rnn_grammar_error']], ax=axs[1])
    g2.set_xticklabels(g2.get_xticklabels())
    g2.set(title="netflix_result")
    
    g3 = sns.boxplot(fliersize=2, width=0.5, showmeans=True, meanprops={"marker":"s","markerfacecolor":"white"},
                data=data3[['gpt2_grammar_error','rnn_grammar_error']], ax=axs[2])
    g3.set_xticklabels(g3.get_xticklabels())
    g3.set(title="lyrics_result")    
    
    fig.suptitle('Graph 2. '+save_name)
    plt.savefig('./text_generation_results/'+'_'.join(save_name.split()))

def plot_results_missp(data1, data2, data3, save_name='Misspelling Rate for all datasets'):
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Adjust the subplot layout parameters
    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    g1 = sns.boxplot(fliersize=2, width=0.5, showmeans=True, meanprops={"marker":"s","markerfacecolor":"white"},
                     data=data1[['gpt2_misspelling','rnn_misspelling']], ax=axs[0])
    g1.set_xticklabels(g1.get_xticklabels())
    g1.set(title="news_result")
    
    g2 = sns.boxplot(fliersize=2, width=0.5, showmeans=True, meanprops={"marker":"s","markerfacecolor":"white"},
                data=data2[['gpt2_misspelling','rnn_misspelling']], ax=axs[1])
    g2.set_xticklabels(g2.get_xticklabels())
    g2.set(title="netflix_result")
    
    g3 = sns.boxplot(fliersize=2, width=0.5, showmeans=True, meanprops={"marker":"s","markerfacecolor":"white"},
                data=data3[['gpt2_misspelling','rnn_misspelling']], ax=axs[2])
    g3.set_xticklabels(g3.get_xticklabels())
    g3.set(title="lyrics_result")    
    
    fig.suptitle('Graph 3. '+save_name)
    plt.savefig('./text_generation_results/'+'_'.join(save_name.split()))
# --------------------------------------------------------------------------------------------
