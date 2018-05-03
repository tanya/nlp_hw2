import os
import io
from numpy.random import randint
try:
    import ujson as json
except:
    import json
try:
    import cPickle as pickle
except:
    import pickle
import warnings

__all__ = ['Vocabulary']

class Vocabulary(object):
    """An implementation that manages the interface between a token dataset and the 
       machine learning algorithm. 
       
    Input:
        name                    name of the vocabulary
        (default=nameless)      default: nameless
                                
        file_type               what kind of saving method (json or pickle currently)
        (default=json)        
                                
        use_mask                assigns first vocab index to mask token; 
        (default=False)         used in some sequence models to handle variable lengths
                                
        emit_unks               the vocabulary will output UNK tokens for 
        (default=False)         out of vocabulary items.
                                
        use_start
        (default=False)
        
        mask_symbol             the symbol used in the vocabulary 
        (default='<MASK>')      note: it's unlikely the symbol will ever actually be seen,
                                    so it is just used as a placeholder
                                
        start_symbol            the symbol used to indicate the beginning of a sentence
        (default='<START>')     If `use_start` is set to True, this will be added to the vocabulary.
        
        end_symbol              the symbol used to indicate the end of a sentence
        (default='<END>')       if `use_start` is set to True, this will be added to the vocabulary.
       
    Starting from the original implementation[1], this has been modified and customized.

    [1] Tim Vieira; https://github.com/timvieira/arsenal
    """

    def __init__(self,  name="nameless", file_type="json", use_mask=False, 
                                                           emit_unks=False, 
                                                           use_start=False,
                                                           mask_symbol='<MASK>', 
                                                           start_symbol='<START>',
                                                           end_symbol='<END>'):

        self.file_type = file_type
        self.name = name
        
        self._mapping = {}   # str -> int
        self._flip = {}      # int -> str; timv: consider using array or list
        self._i = 0
        self._frozen = False
        self._growing = True

        ### mask symbol for use in masked recurrent networks
        ### usually need to be the 0th index
        self.use_mask = use_mask
        self.mask_symbol = mask_symbol
        if self.use_mask:
            self.add(self.mask_symbol)

        ### unk symbol for out of vocabulary tokens
        self.emit_unks = emit_unks
        self.unk_symbol = "<UNK>"
        if self.emit_unks:
            self.add(unk_symbol)

        ### start symbol for sequence models
        self.use_start = use_start
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        if self.use_start:
            self.add(self.start_symbol)
            self.add(self.end_symbol)



    def __repr__(self):
        return 'Vocabulary(size=%s,frozen=%s)' % (len(self), self._frozen)

    def freeze(self, emit_unks=False):
        self.emit_unks = emit_unks
        if emit_unks and self.unk_symbol not in self:
            self.add(self.unk_symbol)
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def stop_growth(self):
        self._growing = False

    @property
    def unk_id(self):
        return self[self.unk_symbol]

    @property
    def mask_id(self):
        return self[self.mask_symbol]

    @classmethod
    def from_iterable(cls, s, *args, **kwargs):
        inst = cls(*args, **kwargs)
        inst.add_many(s)
        return inst

    @classmethod
    def from_nlp_data(cls, iterable):
        ''' ugly api... '''
        vocab = cls()
        vocab.use_mask = True
        vocab.add(vocab.mask_symbol)
        vocab.add(vocab.unk_symbol)
        vocab.add_many(iterable)
        return vocab

    def keyset(self):
        keys = set(self._mapping.keys())
        if self.mask_symbol in keys:
            keys.remove(self.mask_symbol)
        return keys

    def iterkeys(self):
        for k in self._mapping.iterkeys():
            if (k==self.unk_symbol or k==self.mask_symbol):
                continue
            else: 
                yield k

    def fullkeys(self):
        return list(self._mapping.keys())

    def keys(self):
        return [k for k in list(self._mapping.keys()) if (k!=self.unk_symbol and
                                                          k!=self.mask_symbol)]

    ### items
    #### iter items, fullitems and items

    def iteritems(self):
        for k,v in self._mapping.items():
            if k==self.unk_symbol or k==self.mask_symbol:
                continue
            yield k,v

    def fullitems(self):
        return list(self._mapping.items())

    def items(self):
        return [(k,v) for k,v in list(self._mapping.items()) if (k!=self.unk_symbol and
                                                                 k!=self.mask_symbol)]

    def values(self):
        return [v for k,v in list(self._mapping.items()) if (k!=self.unk_symbol and
                                                             k!=self.mask_symbol)]

    def fullvalues(self):
        return list(self._mapping.values())


    def filter_generator(self, seq, emit_none=False):
        """
        Apply Vocabulary to sequence while filtering. By default, `None` is not
        emitted, so please note that the output sequence may have fewer items.
        """
        if emit_none:
            for s in seq:
                yield self[s]
        else:
            for s in seq:
                x = self[s]
                if x is not None:
                    yield x

    def filter(self, seq, *args, **kwargs):
        return list(self.filter_generator(seq, *args, **kwargs))

    def add_many(self, x):
        return [self.add(k) for k in x]

    def lookup(self, i):
        if i is None:
            return None
        return self._flip[i]

    def lookup_many(self, x):
        for k in x:
            yield self.lookup(k)

    def __contains__(self, k):
        return k in self._mapping

    def __getitem__(self, k):
        try:
            return self._mapping[k]
        except KeyError:
            if self._frozen and self.emit_unks:
                return self._mapping[self.unk_symbol]
            elif self._frozen:
                raise ValueError('Vocabulary is frozen. Key "%s" not found.' % (k,))
            elif not self._growing:
                return None
            else:
                x = self._mapping[k] = self._i
                self._i += 1
                self._flip[x] = k
                return x
    add = __getitem__

    def __setitem__(self, k, v):
        assert k not in self._mapping
        if self._frozen: raise ValueError("Vocabulary is frozen. Key '%s' cannot be changed")
        assert isinstance(v, int)
        self._mapping[k] = v
        self._flip[v] = k

    def __iter__(self):
        for i in xrange(len(self)):
            yield self._flip[i]

    def enum(self):
        for i in xrange(len(self)):
            yield (i, self._flip[i])

    def __len__(self):
        return len(self._mapping)

    @classmethod
    def from_config(cls, config):
        data = dict(recursive_tuple_fix(config.pop('data')))
        new_vocab = cls()
        new_vocab.__dict__.update(config)
        for k,v in data.items():
            new_vocab._mapping[k] = v
            new_vocab._flip[v] = k
        new_vocab._i = len(new_vocab) 
        return new_vocab

    @classmethod
    def load(cls, filename, file_type='json'):
        """ config types supported: json, pickle """
        if not os.path.exists(filename):
            warnings.warn("file not found", RuntimeWarning)
            return cls()
        
        if file_type == 'json':
            with io.open(filename, 'r') as fp:
                config = json.load(fp)    
        elif file_type == 'pickle':
            with io.open(filename, 'rb') as fp:
                config = pickle.load(fp)
        else:
            warnings.warn("Configuration type not understood", RuntimeWarning)
            return cls()
        return cls.from_config(config)

    def _config(self):
        config = {"emit_unks": self.emit_unks,
                  "use_mask": self.use_mask,
                  "_frozen": self._frozen,
                  "_growing": self._growing, 
                  "file_type": self.file_type, 
                  "name": self.name}
        return config

    def save(self, filename):
        config = self._config()
        config['data'] = tuple(self._mapping.items())
        if self.file_type == 'json':
            with io.open(filename, 'w', encoding='utf8') as fp:
                json.dump(config, fp)
        elif self.file_type == 'pickle':
            with io.open(filename, 'wb') as fp:
                pickle.dump(config, fp)
        else:
            warnings.warn("Vocabulary {} not saved; unknown save method".format(self.name), 
                          RuntimeWarning)


def recursive_tuple_fix(item):
    if isinstance(item, list):
        return tuple([recursive_tuple_fix(subitem) for subitem in item])
    else:
        return item
