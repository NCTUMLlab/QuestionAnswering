from babi_util import *
from keras.utils.data_utils import get_file

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

def read_single_task(task_indx):
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    tar = tarfile.open(path)
    task_id = 'qa%d'%task_indx
    task_name = challenges[task_id]
    train_stories = get_stories(tar.extractfile(task_name.format('train')))
    test_stories = get_stories(tar.extractfile(task_name.format('test')))
    
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
    vocab_size = len(vocab) + 1
    
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
    
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    return inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, story_maxlen, query_maxlen, vocab_size, word_idx
