try:
    with open("global_setup.py") as setupfile:
        exec(setupfile.read())
except FileNotFoundError:
    print('Setup already completed')


import random
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
from nltk import tokenize # Text to sentences
import pandas as pd # Train and Validation data generation
import re
import pprint
from src.wikipedia import Wikipedia
#random.seed(12345)


wikipedia = Wikipedia(
    language="simple",
    cache_directory_url=False
)

# Cleaning up simple wikipedia texts
pattern_ignored_words = re.compile(
    r"""
    (?:(?:thumb|thumbnail|left|right|\d+px|upright(?:=[0-9\.]+)?)\|)+
    |^\s*\|.+$
    |^REDIRECT\b""",
    flags=re.DOTALL | re.UNICODE | re.VERBOSE | re.MULTILINE)
pattern_new_lines = re.compile('[\n\r ]+', re.UNICODE)
texts = [wikipedia.documents[i].text for i in range(len(wikipedia.documents))]
texts = [pattern_ignored_words.sub('', texts[i]) for i in range(len(texts))]
texts = [pattern_new_lines.sub(' ', texts[i]) for i in range(len(texts))]
texts = [texts[i].replace("\\", "") for i in range(len(texts))]
texts = [texts[i].replace("\xa0", " ") for i in range(len(texts))]


# Simple wikipedia article texts into single sentences

sentences = []
sentences += [tokenize.sent_tokenize(texts[i]) for i in range(len(texts))]
#sentences += [texts[i].split(". ") for i## 6. Divide into sentences in range(len(texts))] #len(texts)
# Now sentences is a list of lists. The next expression flattens it into one long list.
sentences = [item for sublist in sentences for item in sublist]


print(len(sentences))
for i in reversed(range(len(sentences))):
    if len(sentences[i]) < 20 or len(sentences[i]) > 100 \
        or sentences[i][0:9] == "Category:" \
        or sentences[i][0:13] == "Related pages" \
        or sentences[i][0:10] == "References" \
        or sentences[i][0:14] == "Other websites":
        sentences.pop(i)
print(len(sentences))

#Gallery - do something?



## 8. Generate training data
alphabets = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14,
            'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25, 
            '0':26, '1':27, '2':28, '3':29, '4':30, '5':31, '6':32, '7':33, '8':34, '9':35, 
            ' ':36, ',':37, '.':38, ':':39, ';':40, '"':41, "'":42, '':43, '(':44, ')':45} #43 = unknown symbol

idxs = [alphabets[ch] if ch in alphabets else 43 for ch in 'az 123#']

idxs

#one_hot = tf.one_hot(idxs, depth=len(alphabets), dtype=tf.uint8)

#sess = tf.InteractiveSession()
#one_hot.eval()
one_hot = to_categorical(idxs, num_classes = len(alphabets))
one_hot



sentences_idxs = []
for i in range(len(sentences)):
    idx = []
    for j in sentences[i]:
        if j in alphabets:
            idx += [alphabets[j]]
        else:
            idx += [43]
    sentences_idxs.append(idx)
    
#sentences_onehot = [tf.one_hot(sentences_idxs[i], depth=len(alphabets), dtype=tf.uint8) for i in range(len(sentences_idxs))]



#sentences_onehot = [tf.one_hot(sentences_idxs[i], depth=len(alphabets), dtype=tf.uint8) for i in range(10000)]
sentences_onehot = [to_categorical(sentences_idxs[i], num_classes = len(alphabets)) for i in range(100000)]
sentences = sentences[0:100000]




# Generate the data examples
# X and Y are identical for the test purposes

data = pd.DataFrame(
    {'X': sentences_onehot,
     'Y': sentences
    })

print(len(sentences_onehot[100][0]))
print(len(sentences_onehot))




#####################################################

import os

import keras
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, Nadam

#from KerasDeepSpeech.data import combine_all_wavs_and_trans_from_csvs
from KerasDeepSpeech.generator import BatchGenerator
from KerasDeepSpeech.model import *
from KerasDeepSpeech.report import ReportCallback
from KerasDeepSpeech.utils import load_model_checkpoint, save_model, MemoryCallback

#####################################################


#######################################################

# Prevent pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#######################################################


def main(args):
    '''
    There are 5 simple steps to this program
    '''

    #1. combine all data into 2 dataframes (train, valid)
    print("Getting data from arguments")
    #train_dataprops, df_train = combine_all_wavs_and_trans_from_csvs(args.train_files, sortagrad=args.sortagrad)
    #valid_dataprops, df_valid = combine_all_wavs_and_trans_from_csvs(args.valid_files, sortagrad=args.sortagrad)

    train_ratio = 0.9 #90% of data used for training, 10% for validation
    args.model_arch = 0
    args.opt = "adam"
    args.train_steps = 0
    args.epochs = 10
    args.valid_steps = 0
    args.batchsize = 32 #was 16
    args.name = ""
    args.loadcheckpointpath = ""
    args.fc_size = 512
    args.rnn_size = 512
    args.learning_rate = 0.01
    args.memcheck = False
    args.tensorboard = True
    
    model_input_type = "text"
    
    
    df_train = data[0:int(train_ratio * len(sentences_onehot))]
    df_valid = data[int(train_ratio * len(sentences_onehot)):]


    ## 2. init data generators
    print("Creating data batch generators")
    traindata = BatchGenerator(dataframe=df_train, dataproperties=None,
                              training=True, batch_size=args.batchsize, model_input_type=model_input_type)
    validdata = BatchGenerator(dataframe=df_valid, dataproperties=None,
                              training=False, batch_size=args.batchsize, model_input_type=model_input_type)




    output_dir = os.path.join('checkpoints/results',
                                  'model%s_%s' % (args.model_arch,
                                             args.name))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    ## 3. Load existing or create new model
    if args.loadcheckpointpath:
        # load existing
        print("Loading model")

        cp = args.loadcheckpointpath
        assert(os.path.isdir(cp))

        model_path = os.path.join(cp, "model")
        # assert(os.path.isfile(model_path))

        model = load_model_checkpoint(model_path)


        print("Model loaded")
    else:
        # new model recipes here
        print('New model DS{}'.format(args.model_arch))
        if (args.model_arch == 0):
            # DeepSpeech1 with Dropout
            model = ds1_dropout(input_dim=len(alphabets), fc_size=args.fc_size, rnn_size=args.rnn_size,dropout=[0.1,0.1,0.1], output_dim=len(alphabets) + 1)

        elif(args.model_arch==1):
            # DeepSpeech1 - no dropout
            model = ds1(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)

        elif(args.model_arch==2):
            # DeepSpeech2 model
            model = ds2_gru_model(input_dim=161, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)

        elif(args.model_arch==3):
            # own model
            model = ownModel(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, dropout=[0.1, 0.1, 0.1], output_dim=29)

        elif(args.model_arch==4):
            # graves model
            model = graves(input_dim=26, rnn_size=args.rnn_size, output_dim=29, std=0.5)

        elif(args.model_arch==5):
            #cnn city
            model = cnn_city(input_dim=161, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)

        elif(args.model_arch == 6):
            # constrained model
            model = const(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)
        else:
            raise("model not found")

        print(model.summary(line_length=80))

        #required to save the JSON
        save_model(model, output_dir)

    if (args.opt.lower() == 'sgd'):
        opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    elif (args.opt.lower() == 'adam'):
        opt = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    elif (args.opt.lower() == 'nadam'):
        opt = Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    else:
        raise "optimiser not recognised"

    model.compile(optimizer=opt, loss=ctc)

    ## 4. train

    if args.train_steps == 0:
        args.train_steps = len(df_train.index) // args.batchsize
        # print(args.train_steps)
    # we use 1/xth of the validation data at each epoch end to test val score
    if args.valid_steps == 0:

        args.valid_steps = (len(df_valid.index) // args.batchsize)
        # print(args.valid_steps)


    if args.memcheck:
        cb_list = [MemoryCallback()]
    else:
        cb_list = []

    if args.tensorboard:
        tb_cb = TensorBoard(log_dir='./tensorboard/{}/'.format(args.name), write_graph=False, write_images=True)
        cb_list.append(tb_cb)

    y_pred = model.get_layer('ctc').input[0]
    input_data = model.get_layer('the_input').input

    report = K.function([input_data, K.learning_phase()], [y_pred])
    report_cb = ReportCallback(report, validdata, model, args.name, save=True)

    cb_list.append(report_cb)

    model.fit_generator(generator=traindata.next_batch(),
                        steps_per_epoch=args.train_steps,
                        epochs=args.epochs,
                        callbacks=cb_list,
                        validation_data=validdata.next_batch(),
                        validation_steps=args.valid_steps,
                        initial_epoch=0,
                        verbose=1,
                        class_weight=None,
                        max_q_size=10,
                        workers=1,
                        pickle_safe=False
                        )

    # K.clear_session()

    ## These are the most important metrics
    print("Mean WER   :", report_cb.mean_wer_log)
    print("Mean LER   :", report_cb.mean_ler_log)
    print("NormMeanLER:", report_cb.norm_mean_ler_log)

    # export to csv?
    K.clear_session()





class Object(object):
    pass

args = Object()



main(args)
