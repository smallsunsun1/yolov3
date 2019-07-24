import glob
import os
import pickle
import shutil

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

DEBUG = False

def echo(*params):
    if DEBUG:
        print(*params)

class P():
    path_vid = '../data/mlf_vid.txt'
    path_url = '../data/mlf_urls.txt'

    frame_dir = '../frames'
    path_frames = glob.glob(os.path.join(frame_dir, '*', '*.jpg'))

    dir_result = '../result'
    path_result = os.path.join(dir_result, 'res.pkl')

# directory operation
def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# txt
def load_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [d.strip() for d in f.readlines()]
    return data

def write_txt(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

# pickle
def load_pkl(filename):
    return pickle.load(open(filename, 'rb'))

def dump_pkl(data, filename):
    pickle.dump(data, open(filename, 'wb'), protocol=4)

# function
def para_func(func, params, cpu_task=True):
    if cpu_task:
        pool = Pool()
    else:
        pool = ThreadPool()
    if isinstance(params[0], (list, tuple)):
        res = pool.starmap(func, params)
    else:
        res = pool.map(func, params)
    pool.close()
    pool.join()
    return res

def move(dirname, label, src, wanna_neg=False):
    assert len(label) == len(src)

    pos = os.path.join(dirname, 'pos')
    neg = os.path.join(dirname, 'neg')

    mkdir(pos)
    mkdir(neg)

    def single_move(l, s):
        dirname = pos if l else neg
        dst = os.path.join(dirname, os.path.basename(s))
        if l or wanna_neg:
            shutil.copy(s, dst)
            print('Copying {} to {} ...'.format(s, dst))

    params = list(zip(label, src))
    # print(type(params[0]))
    para_func(single_move, params)


