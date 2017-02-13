
import numpy as np

from collections import defaultdict
import string

TAG2NUM = {
        'B' : 0,
        'I' : 1,
        'E' : 2,
        'O' : 3,
        'S' : 4,
}

NUM2TAG = {
        0 : 'B',
        1 : 'I',
        2 : 'E',
        3 : 'O',
        4 : 'S',
}


def charlookup():
    res = defaultdict(int)
    count = 1

    for ch in string.printable:
        res[ch] = count
        count += 1

    return res, count

CHARLOOKUP, CHARCOUNT = charlookup()

def tag(first, seq, last):
    tag_seq = []
    for i in range(len(seq)):
        prev = seq[i - 1] if i > 0 else first
        current = seq[i]
        nxt = seq[i + 1] if i < len(seq) - 1 else last

        if current.isalnum():
            if prev.isalnum():
                if nxt.isalnum():
                    tag_seq.append('I')
                else:
                    tag_seq.append('E')
            else:
                if nxt.isalnum():
                    tag_seq.append('B')
                else:
                    tag_seq.append('S')
        else:
            tag_seq.append('O')

    return tag_seq

def charas(filename):
    with open(filename, 'r') as f:
        for line in f:
            for ch in line:
                yield ch

class CharData(object):

    def __init__(self, filename):
        self.charas = charas(filename)
        self.end_of_prev = ''
        self.preread = ''

    def next_batch(self, batch_size):
        seq = self.preread
        count = 0
        while count < batch_size:
            ch = next(self.charas)
            if ch:
                seq = seq + ch
                count += 1
            else:
                break

        xs = seq
        self.preread = next(self.charas) or ''

        ys = tag(self.end_of_prev, seq, self.preread)

        self.end_of_prev = seq[-1]

        return xs, ys, len(seq)

class SeqData(object):

    def __init__(self, filename):
        self.chardata = CharData(filename)
        self.eye = np.eye(5)

    def next_batch(self, batch_size):
        x, y, l = self.chardata.next_batch(batch_size)

        nx = np.array([np.array([[CHARLOOKUP[c] for c in x]]).T])
        ny = np.array([self.eye[[TAG2NUM[t] for t in y]]])

        nl = np.array([l])

        return nx, ny, nl

SENTENCE_END = '.:;?!,'

def sentences(chars):
    acc = ''
    for c in chars:
        acc += c
        if c in SENTENCE_END:
            yield acc
            acc = ''


class SentenceBatcher(object):

    def __init__(self, filename, max_len):
        self.sentences = sentences(charas(filename))
        self.eye = np.eye(5)
        self.max_len = max_len

    def next_batch(self, num_s):
        xs = []
        ys = []
        ss = []
        ts = []

        for _ in range(num_s):
            sen = next(self.sentences)
            while len(sen) > self.max_len:
                sen = next(self.sentences)
            if sen is None:
                break

            tag_seq = tag('', sen, '')

            x = np.array([CHARLOOKUP[c] for c in sen])
            y = self.eye[[TAG2NUM[t] for t in tag_seq]]

            pad_x = np.pad(x, (0, self.max_len - x.shape[0]), 'constant', constant_values=(0,0))
            pad_y = np.pad(y, [(0, self.max_len - y.shape[0]), (0,0)], 'constant', constant_values=((0,0), (0,0)))

            pad_x = np.array([[c] for c in pad_x])

            xs.append(pad_x)
            ys.append(pad_y)
            ss.append(sen)
            ts.append(tag_seq)

        xr = np.stack(xs)
        yr = np.stack(ys)

        return xr, yr, ss, ts

def print_inference(ss, ts, labels):
    n = len(ss)

    for i in range(n):
        print(ss[i])
        true_tags = ''.join(ts[i])
        s_len = len(ss[i])
        pred_tags = ''.join([NUM2TAG[labels[i][j]] for j in range(s_len)])

        print(pred_tags)
        print(true_tags)
        print('====================')


if __name__ == "__main__":
    #data = CharData('input.txt')

    #xs, ys, l = data.next_batch(20)

    #print(xs)
    #print(''.join(ys))

    data = SeqData('input.txt')

    xs, ys, l = data.next_batch(10)
    print(xs)
    print(ys)
