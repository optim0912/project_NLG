import fire
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
import torch

PAD, BOS, EOS = 1, 2, 3

class DataLoader():
"""Custom dataloader for learning translator. 
    It has a train and valid dataloader, and a source and target vocabulary. 

    Arguments:
        train_fn : Training set file path(str) except the extention. (ex: train_en --> train)
        valid_fn : Validation set file path(str) except the extention. (ex: valid_en --> valid)
        exts : A tuple containing the extension to path for each language.
        batch_size : Mini batch size for gradient descent. Default = 128
        device : The device containing the batch. Default = 'cpu' (lazy loading)
        max_vocab : The maximum size of the vocabulary, or None for no maximum. Default = 99999999
        max_length : The maximum number of words in a sentence. Default = 255
        fix_length: A fixed length that all examples will be padded to, 
        or None for flexible sequence lengths. Default = None
        use_bos : Use the BOS token.  Default = True
        use_eos : Use the EOS token.  Default = True
        shuffle : Whether to shuffle examples between epochs. Default = True
        dsl : Turn on dual-supervised learning mode. Default = True
        lazy : If true, the dataloader reads the text file one line each time. 
        It prevents memory shortage caused by large datasets, but can cause I/O bottlenecks. default = True
        num_workers : how many subprocesses to use for data loading. 
        0 means that the data will be loaded in the main process. default = 0
        pin_memory :  If True, the data loader will copy Tensors into 
        CUDA pinned memory before returning them. default = False
"""

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False,
                 lazy=True,
                 num_workers=0,
                 pin_memory=False
                 ):

        super(DataLoader, self).__init__()
        
        self.src_init_token, self.src_eos_token = '<bos>' if dsl else None, '<eos>' if dsl else None
        self.tgt_init_token, self.tgt_eos_token = '<bos>' if use_bos else None, '<eos>' if use_eos else None
        self.src_vocab = build_vocab(train_fn, exts[0], max_vocab, self.src_init_token, self.src_eos_token)
        self.tgt_vocab = build_vocab(train_fn, exts[1], max_vocab, self.tgt_init_token, self.tgt_eos_token)

        translate_collate_fn = lambda batch : self.process(batch, fix_length, device)

        if train_fn is not None and valid_fn is not None and exts is not None:
            if lazy:
                train = LazyTranslationDataset(train_fn, exts, max_length)
                valid = LazyTranslationDataset(valid_fn, exts, max_length)
            else:
                train = TranslationDataset(train_fn, exts, max_length)
                valid = TranslationDataset(valid_fn, exts, max_length)
        
            train_sampler = BucketSampler(100 * batch_size, train.sort_key, shuffle=shuffle)
            valid_sampler = BucketSampler(100 * batch_size, train.sort_key, shuffle=False)
            
            self.train_loader = DataLoader(train, batch_size, sampler = train_sampler, 
            num_workers = num_workers, collate_fn = translate_collate_fn, pin_memory = pin_memory)
            self.valid_loader = DataLoader(valid, batch_size, sampler = train_sampler, 
            num_workers = num_workers, collate_fn = translate_collate_fn, pin_memory = pin_memory)
    
    def process(self, minibatch, fix_length, device):
        minibatch = self.pad(minibatch, fix_length)
        minibatch = self.numericalize(minibatch, device)
        return minibatch
    
    def load_vocab(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def numericalize(self, minibatch, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            minibatch ((List[List[str]], List[int]), (List[List[str]], List[int])) :
                Tuple of two tuple of List of tokenized and padded examples 
                and List of lengths of each example.
            device : (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
            
        src_batch, tgt_batch = minibatch
        src_padded, src_lengths = src_batch
        tgt_padded, tgt_lengths = tgt_batch
        
        src_padded_num = [[self.src_vocab.stoi[x] for x in ex] for ex in src_padded]
        tgt_padded_num = [[self.tgt_vocab.stoi[x] for x in ex] for ex in src_padded]
        src_padded_tensor = torch.tensor(src_padded_num, device=device).contiguous()
        tgt_padded_tensor = torch.tensor(tgt_padded_num, device=device).contiguous()
        src_lengths_tensor = torch.tensor(src_lengths, device=device).contiguous()
        tgt_lengths_tensor = torch.tensor(src_lengths, device=device).contiguous()
        
        src_examples = (src_padded_tensor, src_lengths_tensor)
        tgt_examples = (tgt_padded_tensor, tgt_lengths_tensor)
        
        return Batch(src_examples, tgt_examples)
    
    def pad(self, minibatch, fix_length):
        """Pad a batch of examples.
        """
        
        src_minibatch, tgt_minibatch = zip(*minibatch)
        
        if fix_length is None:
            src_max_len = max(len(x) for x in src_minibatch)
            tgt_max_len = max(len(x) for x in tgt_minibatch)
        else:
            src_max_len = fix_length + (self.src_init_token, self.src_eos_token).count(None) - 2
            tgt_max_len = fix_length + (self.tgt_init_token, self.tgt_eos_token).count(None) - 2
            
        src_padded, src_lengths = [], []
        for x in src_minibatch:
            src_padded.append(
                ([] if self.src_init_token is None else [self.src_init_token])
                + x[:max_len]
                + ([] if self.src_eos_token is None else [self.src_eos_token])
                + ["<pad>"] * max(0, src_max_len - len(x)))
            src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(x)))
            
        tgt_padded, tgt_lengths = [], []
        for x in tgt_minibatch:
            tgt_padded.append(
                ([] if self.tgt_init_token is None else [self.tgt_init_token])
                + x[:max_len]
                + ([] if self.tgt_eos_token is None else [self.tgt_eos_token])
                + ["<pad>"] * max(0, tgt_max_len - len(x)))
            tgt_lengths.append(len(tgt_padded[-1]) - max(0, tgt_max_len - len(x)))
            
        return ((src_padded, src_lengths), (tgt_padded, tgt_lengths))
    
    @staticmethod
    def build_vocab(train_fn, ext, max_vocab, init_token, eos_token):
        """Construct the Vocab object from dataset.

        Arguments:
            train_fn : Training set file path(str) except the extention. (ex: train_en --> train)
            ext : Extension to path for a language
            max_vocab : The maximum size of the vocabulary
            init_token : BOS token
            eos_token : EOS token
        """
        src_counter = Counter()
        trg_counter = Counter()
                
        path = Path(train_fn + '_' + ext)
        
        with path.open(encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.strip()
                if line != '':
                    sentence = line.split()
                    counter.update(sentence)
                    
        specials = list(OrderedDict.fromkeys(
            tok for tok in ["<unk>", "<pad>", init_token, eos_token] 
            if tok is not None))
        return Vocab(counter, max_vocab, specials=specials)

class TranslationDataset(Dataset):
    """Defines a dataset for machine translation. 
    Create a TranslationDataset given path.

    Arguments:
        path: Common prefix(str) of paths to the data files for both languages.
        exts: A tuple containing the extension to path for each language.
        max_length : The maximum number of words in a sentence.
    """
        
    def __init__(self, path, exts, max_length):
        
        self.src_path, self.trg_path = tuple(Path(path + '_' + x) for x in exts)
        self.src_sentences = []
        self.trg_sentences = []
        self.sort_key = []
        self.len_dataset = 0
        
        with self.src_path.open(encoding='utf-8') as src_file, self.trg_path.open(encoding='utf-8') as trg_file:
            while True:
                src_line, trg_line = src_file.readline(), trg_file.readline()
                if src_line == '' and trg_line == '':
                    break
                src_line, trg_line = src_line.strip(), trg_line.strip()
                src_length, trg_length = len(src_line.split()), len(trg_line.split())
                if max_length and max_length < max(src_length, trg_length):
                    continue
                if src_line != '' and trg_line != '':
                    self.src_sentences.append(src_line)
                    self.tgt_sentences.append(src_line)
                    self.sort_key.append(trg_length + max_length * src_length)
                    self.len_dataset += 1
    
    def __getitem__(self, index):
        src_sentence = self.src_sentences[index]
        trg_sentence = self.trg_sentences[index]

        return (src_sentence, trg_sentence)
        
    def __len__(self):
        return self.len_dataset
        
class LazyTranslationDataset(Dataset):
    """Defines a dataset for machine translation. 
    Create a TranslationDataset given path. To support large text dataset, 
    lazy loading method is applied. 

    Arguments:
        path: Common prefix(str) of paths to the data files for both languages.
        exts: A tuple containing the extension to path for each language.
        max_length : The maximum number of words in a sentence.
    """
        
    def __init__(self, path, exts, max_length):
        
        self.src_path, self.trg_path = tuple(Path(path + '_' + x) for x in exts)
        self.src_line_offset = []
        self.trg_line_offset = []
        self.sort_key = []
        self.len_dataset = 0
        
        with self.src_path.open(encoding='utf-8') as src_file, self.trg_path.open(encoding='utf-8') as trg_file:
            while True:
                src_offset, trg_offset = src_file.tell(), trg_file.tell()
                src_line, trg_line = src_file.readline(), trg_file.readline()
                if src_line == '' and trg_line == '':
                    break
                src_line, trg_line = src_line.strip(), trg_line.strip()
                src_length, trg_length = len(src_line.split()), len(trg_line.split())
                if max_length and max_length < max(src_length, trg_length):
                    continue
                if src_line != '' and trg_line != '':
                    self.src_line_offset.append(src_offset)
                    self.trg_line_offset.append(trg_offset)
                    self.sort_key.append(trg_length + max_length * src_length)
                    self.len_dataset += 1
    
    def __getitem__(self, index):
        src_offset = self.src_line_offset[index]
        trg_offset = self.trg_line_offset[index]
        
        with self.src_path.open(encoding='utf-8') as src_file, self.trg_path.open(encoding='utf-8') as trg_file:
            src_file.seek(src_offset)
            trg_file.seek(trg_offset)
            src_line, trg_line = src_file.readline(), trg_file.readline()
            src_line, trg_line = src_line.strip(), trg_line.strip()
            src_sentence, trg_sentence = src_line.split(), trg_line.split()

            return (src_sentence, trg_sentence)
        
    def __len__(self):
        return self.len_dataset

class BucketSampler(DistributedSampler):
    """Custom DistributedSampler for learning translators. 
    Sorting is performed to reduce the number of pads. However, if the entire sorting is performed, 
    the shuffle effect disappears, so sorting is performed only as much as the bucket size. 

    Arguments:
        bucket_size : Sorting is performed only as much as the bucket size.
        sort_key : Sort_key for sorting.
        **kwargs : Arguments of the inheriting DistributedSampler class. 
        (dataset, num_replicas, rank, shuffle(default = True), seed, drop_last)
    """

    def __init__(self, bucket_size, sort_key, **kwargs):
    
        super(DistributedSampler, self).__init__(**kwargs)
        self.bucket_size = bucket_size
        self.sort_key = sort_key
        self.num_buckets = self.num_samples // self.bucket_size
        self.num_extra = self.num_samples % self.bucket_size
    
    def __iter__(self):
    
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g1 = torch.Generator()
            g1.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g1).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        # To sort while maintaining the shuffle effect. 
        sub_indices_list = []
        for bucket_index in range(self.num_buckets):
            start_index = self.bucket_size * bucket_index
            end_index = self.bucket_size * (bucket_index + 1)
            sub_indices = indices[start_index:end_index]
            sub_indices.sort(key = self.sort_key)
            sub_indices_list.append(sub_indices)
            
        if self.num_extra != 0:
            sub_indices = indices[-self.num_extra:]
            sub_indices.sort(key = self.sort_key)
            sub_indices_list.append(sub_indices)
            
        if self.shuffle:
            g2 = torch.Generator()
            temp_indices = torch.randperm(len(sub_indices_list), generator=g2).tolist()
        else:
            indices = list(range(len(sub_indices_list)))

        new_indices = []
        for temp_indice in temp_indices:
            new_indices += sub_indices_list[temp_indice]
            
        return iter(new_indices)
        
class Batch():
    """The class representing an batch of a dataset. 
    It consists of source examples and target examples. 
    """
        
    def __init__(self, src_examples, tgt_examples):
        self.src = src_examples
        self.tgt = tgt_examples

def dataloader_test(train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False,
                 lazy=True,
                 num_workers=0,
                 pin_memory=False
                 ):
    """Test the dataloader. Outputs the vocabulary size and one batch of source and target. 
        Also, the two examples in the batch are converted into strings and displayed. 
    """
    
    loader = DataLoader(train_fn,
                 valid_fn,
                 exts,
                 batch_size,
                 device,
                 max_vocab,
                 max_length,
                 fix_length,
                 use_bos,
                 use_eos,
                 shuffle,
                 dsl,
                 lazy=True,
                 num_workers=0,
                 pin_memory=False
                 )
    
    src_vocab_itos = loader.src_vocab.itos
    tgt_vocab_itos = loader.tgt_vocab.itos
    
    print(f'src_vocab_size : {len(src_vocab_itos}')
    print(f'tgt_vocab_size : {len(tgt_vocab_itos}')

    for batch_index, batch in enumerate(loader.train_loader):
        print('[src_batch]')
        print(batch.src)
        print('')
        
        print('[tgt_batch]')
        print(batch.tgt)
        print('')
        
        for index in range(2):
            src_sentence_int = batch.src[0][index]
            src_sentence_str = ' '.join(list(map(lambda x : src_vocab_itos[x], src_sentence_int)))
            print(src_sentence_str)
            
            tgt_sentence_int = batch.tgt[0][index]
            tgt_sentence_str = ' '.join(list(map(lambda x : tgt_vocab_itos[x], tgt_sentence_int)))
            print(tgt_sentence_str)
            
        if batch_index > 1:
            break

if __name__ == '__main__':
    fire.Fire({'test' : dataloader_test})
