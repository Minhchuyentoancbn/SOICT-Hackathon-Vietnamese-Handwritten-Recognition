import difflib
import pandas as pd
import pickle

from collections import defaultdict
from config import LABEL_FILE

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode) 
        self.smallest_str = None
        self.end = None

    def __getitem__(self, c):
        return self.children[c]


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, s: str):
        node = self.root
        for c in s:
            node = node[c]
            if node.smallest_str is None:
                node.smallest_str = s
        node.end = s

    def get_similar(self, s):
        node = self.root
        for i, c in enumerate(s):
            if c not in node.children:
                i -= 1
                break
            node = node[c]
        return (node.smallest_str or node.end, i + 1)


class Matcher:
    def __init__(self, dic: dict):
        self.trie = Trie()
        for s in dic:
            self.trie.insert(s)

    def get_match(self, s: str) -> tuple:
        return self.trie.get_similar(s)


def trie_correction(texts, dictionary, threshold=0.85):
    preds = []
    match_score = []
    matcher = Matcher(dictionary)
    for query_txt in texts:
        # key, score = matcher.get_match(query_txt.lower())
        key, score = matcher.get_match(query_txt)
        if score > threshold:
            preds.append(key)
            match_score.append(score)
        else:
            preds.append(query_txt)
            match_score.append(0)
    return preds, match_score


def diff_correction(texts, dictionary, threshold=0.85):
    def sentence_distance(p1, p2):
        return difflib.SequenceMatcher(None, p1, p2).ratio()
    
    preds = []
    match_score = []

    for query_txt in texts:
        # dis_list = [(key, sentence_distance(query_txt.lower(), key)) for key in dictionary.keys()]
        dis_list = [(key, sentence_distance(query_txt, key)) for key in dictionary.keys()]
        dis_list = sorted(dis_list,key=lambda tup: tup[1],reverse=True)[:5]
        key, score = dis_list[0]
        if score > threshold:
            preds.append(key)
            match_score.append(score)
        else:
            preds.append(query_txt)
            match_score.append(0)
    return preds, match_score


def get_heuristic_correction(type_='diff'):
    return trie_correction if type_=='trie' else diff_correction


class Correction:
    def __init__(self, dictionary=None, mode="ed", valid=False):
        assert mode in ["trie", "ed"], "Mode is not supported"
        self.mode = mode
        self.dictionary = dictionary

        self.use_trie = False
        self.use_ed = False

        if self.mode == 'trie':
            self.use_trie = True
        if self.mode == 'ed':
            self.use_ed = True
        
        if self.use_ed:
            self.ed = get_heuristic_correction('diff')
        if self.use_trie:
            self.trie = get_heuristic_correction('trie')
        
        if self.use_ed or self.use_trie:
            if self.dictionary is None:
                self.dictionary = {}
                df = pd.read_csv(LABEL_FILE, sep='\t', header=None, encoding='utf-8', na_filter=False)
                if valid:
                    with open('train_inds.pkl', 'rb') as f:
                        train_inds = pickle.load(f)
                    df = df.iloc[train_inds]

                labels = df[1].value_counts()
                self.dictionary = labels.to_dict()
                # for id, row in df.iterrows():
                #     # self.dictionary[row.text.lower()] = row.lbl
                #     self.dictionary[row] = row.lbl

    def __call__(self, query_texts, return_score=False):
        if self.use_ed:
            preds, score = self.ed(query_texts, self.dictionary)
            
        if self.use_trie:
            preds, score = self.trie(query_texts, self.dictionary)
        
        if return_score:
            return preds, score
        else:
            return preds
