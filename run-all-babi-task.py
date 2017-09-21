from __future__ import print_function
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import backend as K
from functools import reduce
from keras.objectives import mse, categorical_crossentropy
import time
from babi_util import *
from QA_Models import *

challenges = {
    # QA1 with 10,000 samples
    'qa1': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'qa2': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    #QA3 with 10,000 samples
    'qa3': 'tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_{}.txt',
    #QA4 with 10,000 samples
    'qa4': 'tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_{}.txt',
    #QA5 with 10,000 samples
    'qa5': 'tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_{}.txt',
    #QA6 with 10,000 samples
    'qa6': 'tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_{}.txt',
    #QA7 with 10,000 samples
    'qa7': 'tasks_1-20_v1-2/en-10k/qa7_counting_{}.txt',
    #QA8 with 10,000 samples
    'qa8': 'tasks_1-20_v1-2/en-10k/qa8_lists-sets_{}.txt',
    #QA9 with 10,000 samples
    'qa9': 'tasks_1-20_v1-2/en-10k/qa9_simple-negation_{}.txt',
    #QA10 with 10,000 samples
    'qa10': 'tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_{}.txt',
    #QA11 with 10,000 samples
    'qa11': 'tasks_1-20_v1-2/en-10k/qa11_basic-coreference_{}.txt',
    #QA12 with 10,000 samples
    'qa12': 'tasks_1-20_v1-2/en-10k/qa12_conjunction_{}.txt', 
    #QA13 with 10,000 samples
    'qa13': 'tasks_1-20_v1-2/en-10k/qa13_compound-coreference_{}.txt',
    #QA14 with 10,000 samples
    'qa14': 'tasks_1-20_v1-2/en-10k/qa14_time-reasoning_{}.txt',
    #QA15 with 10,000 samples
    'qa15': 'tasks_1-20_v1-2/en-10k/qa15_basic-deduction_{}.txt',
    #QA16 with 10,000 samples
    'qa16': 'tasks_1-20_v1-2/en-10k/qa16_basic-induction_{}.txt',
    #QA17 with 10,000 samples
    'qa17': 'tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_{}.txt',
    #QA18 with 10,000 samples
    'qa18': 'tasks_1-20_v1-2/en-10k/qa18_size-reasoning_{}.txt',
    #QA19 with 10,000 samples
    'qa19': 'tasks_1-20_v1-2/en-10k/qa19_path-finding_{}.txt',
    #QA20 with 10,000 samples
    'qa20': 'tasks_1-20_v1-2/en-10k/qa20_agents-motivations_{}.txt'
}

batch_size = 100

def sec2hr(t):
    hr = t/3600
    mins = (t%3600)/60
    sec = t%60
    time_str = '%02d:%02d:%02d'%(hr,mins,sec)
    return time_str


def run_single_task(task_id,outfile):
    task_name = challenges[task_id]
    task_start = time.time()
    print('Running '+task_id)
    train_stories = get_stories(tar.extractfile(task_name.format('train')))
    test_stories = get_stories(tar.extractfile(task_name.format('test')))
    train_stories_sup = get_stories(tar.extractfile(task_name.format('train')),only_supporting=True)
    test_stories_sup = get_stories(tar.extractfile(task_name.format('test')),only_supporting=True)
    
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
    # Reserve 0 for masking via pad_sequences
    vocab_sup = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories_sup +test_stories_sup)))

    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    story_maxlen_sup = max(map(len, (x for x, _, _ in train_stories_sup + test_stories_sup)))
    query_maxlen_sup = max(map(len, (x for _, x, _ in train_stories_sup + test_stories_sup)))
    
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    inputs_train_sup, _,_ = vectorize_stories(train_stories_sup, word_idx, story_maxlen_sup, query_maxlen_sup)
    inputs_test_sup, _, _ = vectorize_stories(test_stories_sup, word_idx, story_maxlen_sup, query_maxlen_sup)
    
    inputs_train_sup_onehot = np.zeros([10000,story_maxlen_sup,vocab_size])
    for i in xrange(10000):
        inputs_train_sup_onehot[i]=np_utils.to_categorical(inputs_train_sup[i],vocab_size)
    
    inputs_test_sup_onehot = np.zeros([1000,story_maxlen_sup,vocab_size])
    for i in xrange(1000):
        inputs_test_sup_onehot[i]=np_utils.to_categorical(inputs_test_sup[i],vocab_size)
        

    prior_model,VMemNN = varMem(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,64,vocab_size)
    earlyStop = EarlyStopping(monitor = 'loss',patience=5,mode='min') 
    prior_model.fit([inputs_train, queries_train],inputs_train_sup_onehot, epochs=20, batch_size=batch_size,verbose=2)
    hist=VMemNN.fit([inputs_train, queries_train], [inputs_train_sup_onehot,answers_train],
               batch_size=batch_size,
               epochs=70,
               callbacks=[earlyStop],
               validation_data=([inputs_test, queries_test], [inputs_test_sup_onehot,answers_test]),
               verbose=2)
    task_end  = time.time()
    due_time  = sec2hr(task_end-task_start)
    acc_train = hist.history['pred_y_acc']
    acc_valid = hist.history['val_pred_y_acc']
    result='Task ID:'+task_id+' Training accuracy:%.4f, Test accuracy:%.4f, nb_epoch: %d, Cost Time:%s\n'%(acc_train[-1],acc_valid[-1],len(acc_train),due_time)
    outfile.write(result)
    outfile.flush()
    print('*'*120)
    print('\n'+result)
    print('*'*120)


path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)
#tasks=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
tasks=[1]
with open('RESULTS.txt','w') as f:
    start_time = time.time()
    for i in tasks:
        task_id = 'qa%d'%(i)
        run_single_task(task_id,f)
    end_time = time.time()
    total_time = sec2hr(end_time-start_time)
    f.write('Total Cost Time: %s'%total_time)
    print('Total Cost Time: %s'%total_time)
