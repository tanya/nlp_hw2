# tagtools.py
# Matthew Stone, CS 533, Spring 2017
# This code contains helper routines so you can get started
# writing custom taggers for data sets like our
# bibliography and ingredients data sets.
# For the key ideas about working with these functions,
# I'd recommend looking at the homework assignment text
# itself, together with the demo notebook designed to get
# you started.  This file is minimally commented,
# with just the bare bones needed to work out what's
# going on for yourself.

import xml.dom.minidom
import os
import nltk
import re
import itertools
import functools
import vocabulary
import numpy as np
import scipy
import sklearn
import random
import heapq
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

def gmap(genf, arglist) :
    '''common iteration pattern where you map a function
       that turns its argument into an interator
       over a list of items and yield all the
       associated items in order'''
    for o in itertools.chain(*itertools.imap(genf, arglist)) :
        yield o

def agrees(s1, s2, partial=True) :
    '''returns true if all the tags in s1 and s2 are the same.
    if partial is true, treat None elements in s1 and s2 as
    wildcards and act as though s1 and s2 were padded with Nones.'''
    for t1 in s1 :
        try :
            t2 = next(s2)
            if t1 == t2 :
                continue
            elif not partial or (t1 and t2) :
                return False
        except StopIteration:
            return partial
    try:
        t2 = next(s2)
        return partial
    except StopIteration:
        return True

def visualize(items, tagging_dict) :
    '''
    takes a list of strings, corresponding to the items in the sequence
    that you're tagging, and a dictionary of descriptions of these items,
    including a list of tags 'predicted', and a list of tags 'actual', as
    well as others (depending on the decoding algorithm) and presents
    an interpretable table so a person can inspect the tagging results
    '''
    if 'actual' not in tagging_dict or 'predicted' not in tagging_dict :
        print ("No actual or predicted results")
        return
    if agrees(iter(tagging_dict['actual']),
              iter(tagging_dict['predicted']),
              partial=False) :
        print ("Analyzed correctly.")
    elif agrees(iter(tagging_dict['actual']),
                iter(tagging_dict['predicted']),
                partial=True) :
        print ("Analyzed partially.")
    else:
        print ("Some errors.")
    for i, w in enumerate(items) :
        specs = [ u"\t{:10} {:10}".format(k, d[i]) for k, d in tagging_dict.items() ]
        print ("{:15}".format(w), "".join(specs))

def bieso_classification_report(y_true, y_pred):
    """
    Classification report for BIO-encoded data.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(y_true)
    y_pred_combined = lb.transform(y_pred)
        
    tagset = set(lb.classes_) 
    tagset = sorted(tagset, key=lambda tag: tag[3:])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
                    
def tagged_tokens(itemDOM) :
    '''
    limit ourselves to ascii characters,
    then return the ntlk tokens from a DOM element tagged with DOM type
    ''' 
    for n in itemDOM.childNodes :
        tag = n.nodeName
        if len(n.childNodes) > 0 and n.childNodes[0].nodeValue :
            line = re.sub(r'[^\x00-\x7F]+', ' ', n.childNodes[0].nodeValue)
            words = nltk.word_tokenize(line) 
            for w in words :
                yield w, tag
                
def bies_tagged_tokens(itemDOM) :
    '''
    limit ourselves to ascii characters,
    then return the ntlk tokens from a DOM element
    tagged with DOM type plus B/I/E/S tags
    ''' 
    for n in itemDOM.childNodes :
        tag = n.nodeName
        if len(n.childNodes) > 0 and n.childNodes[0].nodeValue :
            line = re.sub(r'[^\x00-\x7F]+', ' ', n.childNodes[0].nodeValue)
            words = nltk.word_tokenize(line) 
            for i in range(len(words)) :
                if i == 0 and len(words) == 1 :
                    yield (words[i], u"s: " + tag)
                elif i == 0 :
                    yield (words[i], u"b: " + tag)
                elif i == len(words)-1 :
                    yield (words[i], u"e: " + tag)
                else :
                    yield (words[i], u"i: " + tag)

def all_untagged_tokens(xml_file, keyword) :
    '''
    get all the token strings associated with an xml file.
    useful if you want to compile a vocabulary
    and set up word embedding features
    '''
    eltDOM = xml.dom.minidom.parse(xml_file).documentElement
    gmap(lambda s : s[0],
         gmap(tagged_tokens,
              eltDOM.getElementsByTagName(keyword)))

def build_sparse_embedding(vocab, glovefile, d) :
    '''build an embedding matrix using the passed vocabulary,
    using the glove dataset stored in glovefile
    assuming word vectors in the dataset have dimension d.
    glove tokens are all lower case, so we'll try to match
    lower case, capitalized and upper case versions in the
    vocabulary.  the expectation is that most of the elements
    in the vocabulary are weird (names and other quirky tokens)
    so we'll return a scipy CSR sparse matrix.'''
    
    remaining_vocab = vocab.keyset()
    embeddings = np.zeros((len(remaining_vocab), d))
    
    with open(glovefile, encoding="utf-8") as glovedata :
        fileiter = glovedata.readlines()
        rows = []
        columns = []
        values = []
        
        for line in fileiter :
            line = line.replace("\n","").split(" ")
            try:
                glove_key, nums = line[0], [float(x.strip()) for x in line[1:]]
                for word in (glove_key, glove_key.capitalize(), glove_key.upper()) :
                    if word in remaining_vocab :
                        columns.append(np.arange(len(nums)))
                        rows.append(np.full(len(nums), vocab[word]))
                        values.append(np.array(nums))
                        remaining_vocab.remove(word)
            except Exception as e:
                print("{} broke. exception: {}. line: {}.".format(word, e, x))

        print("{} words were not in glove".format(len(remaining_vocab)))
        return scipy.sparse.coo_matrix((np.concatenate(values),
                                        (np.concatenate(rows),
                                         np.concatenate(columns))),
                                        shape=(len(vocab), d)).tocsr()

def save_sparse_csr(filename, array):
    'helper routine to efficiently save scipy CSR matrix'
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    'helper routine to efficiently load scipy CSR matrix'
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                   shape = loader['shape'])

class DataError(Exception) :
    '''Class for error in reading XML tagger problem
    instances from the file system.'''
    def __init__(self, message) :
        self.message = message
        
class DataManager(object) :
    '''Class for managing the mapping from tagger XML files in the file system
    to feature matrix representations as used for machine learning prediction'''
    
    def __init__(self,
                 train_xml_file, test_xml_file, dev_xml_file,
                 item_keyword,
                 tokenizer, token_view,
                 make_features, get_features) :
        self.intialized = False
        self.train_xml_file = train_xml_file
        self.test_xml_file = test_xml_file
        self.dev_xml_file = dev_xml_file
        self.item_keyword = item_keyword
        self.tokenizer = tokenizer
        self.token_view = token_view
        self.make_features = make_features
        self.get_features = get_features
        self.features = None

    def initialize(self, shuffle=True) :
        '''intialize: query the file system to get the instances
        to work with and use the passed callback to compile the
        features to classify with'''
        
        eltDOM = xml.dom.minidom.parse(self.train_xml_file).documentElement
        self.train_items = [a for a in eltDOM.getElementsByTagName(self.item_keyword)]
        if shuffle :
            np.random.shuffle(self.train_items)
            
        eltDOM = xml.dom.minidom.parse(self.test_xml_file).documentElement
        self.test_items = [a for a in eltDOM.getElementsByTagName(self.item_keyword)]
        if shuffle :
            np.random.shuffle(self.test_items)

        eltDOM = xml.dom.minidom.parse(self.dev_xml_file).documentElement
        self.dev_items = [a for a in eltDOM.getElementsByTagName(self.item_keyword)]
        if shuffle :
            np.random.shuffle(self.dev_items)

        self.initialized = True
        self.features = self.make_features(self)
    
    def all_train_tokens(self) :
        '''iterate over the training tokens associated with the passed data set'''
        
        if self.initialized :
            return gmap(self.tokenizer, self.train_items)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def _data_to_matrices(self, data) :
        '''
        helper function for compiling a feature matrix from a list of DOM objects.
        each row in the matrix corresponds to a token from one of the objects.
        the matrix is returned in sparse CSR representation.
        the tokenize attribute of the data manager is used to break the object up
        and the get_features attribute of the data manager is used
        to get the feature representation from the tokens.
        returns three things:
        - a matrix of features
        - an array of tags
        - an array of starting indices for each of the texts in data
        '''

        row = 0
        rows = []
        columns = []
        values = []
        tags = []
        d_starts = []
        for d in data:
            d_starts.append(row)
            for item, tag in self.tokenizer(d) :
                f = self.get_features(self.features, item)
                columns.append(f)
                rows.append(np.full(f.shape, row, dtype=np.int))
                values.append(np.full(f.shape, 1., dtype=np.float))
                tags.append(tag)
                row = row + 1
        cd = np.concatenate(columns)
        d_starts.append(row)
        return (scipy.sparse.coo_matrix((np.concatenate(values), 
                                         (np.concatenate(rows), 
                                         cd)),  
                                         shape=(row,len(self.features))).tocsr(),
                np.array(tags), np.array(d_starts))

    def dev_item_token_views(self, item_number) :
        '''
        get the list of string representations of the tokens in
        item number item_number from the dev set
        (in preparation for visualizing tagging results)
        '''
        if self.initialized :
            return [self.token_view(item)
                    for item, tag in self.tokenizer(self.dev_items[item_number])]
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def test_features_dev_item(self, item_number) :
        '''
        process the features associated with example item_number from the
        dev set and print a visualization of the true features for each item
        '''
        if self.initialized :
            data_point_slice = self.dev_items[item_number:item_number+1]
            X, y, _ = self._data_to_matrices(data_point_slice)
            tv = self.dev_item_token_views(item_number)
            for i, t in enumerate(tv) :
                print (u"{} ({})".format(t, y[i]))
                line = []
                for j in range(X.indptr[i], X.indptr[i+1]) :
                    line.append(u"\t{}".format(self.features.lookup(X.indices[j])))
                    if (len(line) % 5) == 0 :
                        print (u"".join(line))
                        line = []
                if line :
                    print (u"".join(line))
        else :
            raise DataError("Must call initialized() on Data Manager before accessing data")

    def training_data(self) :
        '''get matrices for the training data and training categories from the collection'''
        if self.initialized :
            return self._data_to_matrices(self.train_items)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def dev_data(self) :
        '''get matrices for the development data and development categories from the collection'''
        if self.initialized :
            return self._data_to_matrices(self.dev_items)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def test_data(self) :
        '''get matrices for the test data and test categories from the collection'''
        if self.initialized :
            return self._data_to_matrices(self.test_items)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")
    

class BeamAStarQ(object) :
    """Priority Queue that distinguishes states and nodes,
       keeps track of priority as cost plus heuristic,
       enables changing the node and priority associated with a state,
       and keeps a record of explored states.
       Maintains a limited horizon at each level,
       limiting search to nodes within a fixed bound of the optimal encountered
       """
    
    # state label for zombie nodes in the queue
    REMOVED = '<removed-state>'     

    # sequence count lets us keep heap order stable
    # despite changes to mutable objects 
    counter = itertools.count()     
    
    @functools.total_ordering
    class QEntry(object) :
        """BeamAStarQ.QEntry objects package together
        state, node, cost, heuristic and priority
        and enable the queue comparison operations"""

        def __init__(self, state, node, cost, heuristic) :
            self.state = state
            self.node = node
            self.cost = cost
            self.heuristic = heuristic
            self.priority = cost + heuristic
            self.sequence = next(BeamAStarQ.counter)
        
        def __le__(self, other) :
            return ((self.priority, self.sequence) <= 
                    (other.priority, other.sequence))
        
        def __eq__(self, other) :
            return ((self.priority, self.sequence) == 
                    (other.priority, other.sequence))
   
    def __init__(self, keyf, width=5.) :
        """
        Set up a new problem with empty queue and nothing explored.
        The key function maps states to integers; beam search
        enforces limits on search as a function of the key.
        The width is the maximum detour allowed in log probability.
        For example 5 means that we won't consider a state that is
        less than 1/32 as likely as the best state encountered
        for a particular key value.
        """
        self.pq = {}   
        self.state_info = {}
        self.keyf = keyf
        self.width = width
        self.best = {}

    def add_node(self, state, node, cost, heuristic):
        """
        Add a new state or update the priority of an existing state
        Returns outcome (added or not) for visualization
        """        
        if state in self.state_info:
            already = self.state_info[state].priority
            if already <= cost + heuristic :
                return False
            self.remove_state_entry(state)
        entry = BeamAStarQ.QEntry(state, node, cost, heuristic)
        self.state_info[state] = entry
        i = self.keyf(state)
        if i not in self.pq :
            self.pq[i] = []
            self.best[i] = None
        heapq.heappush(self.pq[self.keyf(state)], entry)
        return True

    def remove_state_entry(self, state):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.state_info.pop(state)
        entry.state = BeamAStarQ.REMOVED

    def pop_node(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while True:
            alts = [(i, q[0].priority) for i, q in self.pq.items() 
                    if len(q) != 0 and (not self.best[i] or
                                        q[0].priority <= self.best[i] + self.width)]
            if not alts:
                raise KeyError('pop from an empty priority queue')
            take = min(alts, key=lambda x:x[1])[0]
            entry = heapq.heappop(self.pq[take])
            if not self.best[take] :
                self.best[take] = entry.priority
            if entry.state is not BeamAStarQ.REMOVED:
                return entry

def unpack_tuples(t) :
    'helper function to roll out history representation into a list'
    result = []
    while t :
        (a, t) = t
        result.append(a)
    result.reverse()
    return result

def compute_heuristics(log_predictions) :
    '''
    transform predictions into heuristics
    by adding up least costs that must necessarily be incurred
    in analyzing the rest of the string
    '''
    best_nlog_probs = np.max(log_predictions, axis=1)
    values = [0., 0.]
    snlp = 0.
    for j in range(len(best_nlog_probs)-1,-1,-1) :
        snlp -= best_nlog_probs[j] 
        values.append(snlp)
    values.reverse()
    return values

class BeamDecoder(object) :
    '''
    Class for using beam search to find a consistent tag sequence
    analysis based on class probabilities, a model of valid tags,
    and exception handling for model failure.
    '''
    
    def __init__(self, start_status, is_consistent, next_status, do_special) :
        '''
        Set up the beam decoder.  Depends on four parameters:
        - start_status: function that returns the consistency status
        for the start state in search
        - is_consistent: function that says whether it's ever possible
        to transition from tag t1 to tag t2
        - next_status: function that computes the consistency status
        model from the current status, the current tag and the new tag
        (and returns None if this can't be done)
        - do_special: function that recognizes the potential for
        model failure and returns possible ways to salvage the current
        search status to give a partial solution
        '''
        self.start_status = start_status
        self.is_consistent = is_consistent
        self.next_status = next_status
        self.do_special = do_special

    def search(self, tagset, log_predictions, verbose=False) :
        '''
        Given the passed tag set and the passed prediction matrix,
        return the optimal tag sequence and its overall log probability,
        while enforcing the consistency constraints built into
        the decoding model.
        '''

        queue = BeamAStarQ(lambda x: x[0])
        heuristics = compute_heuristics(log_predictions)
    
        def loud_pop() :
            entry = queue.pop_node()
            print ("looking at", entry.state[0], entry.state[1], entry.priority)
            return entry
        def loud_add(i, t, s, n, c) :
            did = queue.add_node((i, t, s), (t, n), c, heuristics[i+1])
            if did :
                print ("added node for", i, t, c + heuristics[i+1])
            else :
                print ("redundant node for", i, t, c + heuristics[i+1])
            
        if verbose :
            qpop, qadd = loud_pop, loud_add
        else :
            qpop, qadd = (queue.pop_node, 
                          lambda i,t,s,n,c: queue.add_node((i,t,s),(t,n),c, heuristics[i+1]))

        qadd(-1, 'START', self.start_status(), None, 0.)
        while True:
            entry = qpop()
            j, t1, status = entry.state
            if j == log_predictions.shape[0] :
                return unpack_tuples(entry.node), entry.cost

            if j+1 < log_predictions.shape[0] :
                for i, t2 in enumerate(tagset) :
                    if self.is_consistent(t1, t2):
                        ns = self.next_status(status, t1, t2)
                        if ns :
                            cost = entry.cost - log_predictions[j+1][i]
                            qadd(j+1, t2, ns, entry.node, cost)
                    specials = self.do_special(status, t1, j+1, entry.node, heuristics) 
                    for node, cost in specials :
                        qadd(log_predictions.shape[0], 'END', status, node, 
                             entry.cost + cost)
            else :
                if self.is_consistent(t1, 'END') :
                    qadd(j+1, 'END', status, entry.node, entry.cost)
