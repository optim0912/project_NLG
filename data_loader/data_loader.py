import fire
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
import torch
from multiprocessing import Pool
import time
from typing import List, Set, Dict, Tuple
from math import ceil

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils import memory_usage

PAD, BOS, EOS = 1, 2, 3

class TranslationDataLoader():
    """Custom dataloader for learning translator. 
        It has a train and valid dataloader, and a source and target vocabulary. 
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
                 pin_memory=False,
                 num_parallel_reads=2
                 ):
        
        self.src_init_token, self.src_eos_token = '<bos>' if dsl else None, '<eos>' if dsl else None
        self.tgt_init_token, self.tgt_eos_token = '<bos>' if use_bos else None, '<eos>' if use_eos else None
        translate_collate_fn = lambda batch : self.process(batch, fix_length, device)

        if train_fn is not None and valid_fn is not None and exts is not None:
            if lazy:
                train = LazyTranslationDataset(train_fn, exts, max_length, num_parallel_reads)
                valid = LazyTranslationDataset(valid_fn, exts, max_length, num_parallel_reads)
            else:
                train = TranslationDataset(train_fn, exts, max_length, num_parallel_reads)
                valid = TranslationDataset(valid_fn, exts, max_length, num_parallel_reads)

            self.src_vocab = self.build_vocab(train.src_counters, max_vocab, 
                              self.src_init_token, self.src_eos_token)
            self.tgt_vocab = self.build_vocab(train.trg_counters, max_vocab, 
                              self.tgt_init_token, self.tgt_eos_token)

            self.train_sampler = BucketSampler(100*batch_size, batch_size, train.sort_key, 
                            dataset=train, shuffle=shuffle, num_replicas=1, rank=0, seed=2021)
            self.valid_sampler = BucketSampler(100*batch_size, batch_size, train.sort_key, 
                            dataset=valid, shuffle=False, num_replicas=1, rank=0)
            
            self.train_loader = DataLoader(train, batch_size, sampler=self.train_sampler, 
            num_workers=num_workers, collate_fn=translate_collate_fn, pin_memory=pin_memory)
            self.valid_loader = DataLoader(valid, batch_size, sampler = self.valid_sampler, 
            num_workers=num_workers, collate_fn=translate_collate_fn, pin_memory=pin_memory)
    
    def process(self, minibatch, fix_length, device):
        minibatch = self.pad(minibatch, fix_length)
        minibatch = self.numericalize(minibatch, device)
        return minibatch
    
    def load_vocab(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.valid_sampler.set_epoch(epoch)
    
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
        tgt_padded_num = [[self.tgt_vocab.stoi[x] for x in ex] for ex in tgt_padded]
        src_padded_tensor = torch.tensor(src_padded_num, device=device).contiguous()
        tgt_padded_tensor = torch.tensor(tgt_padded_num, device=device).contiguous()
        src_lengths_tensor = torch.tensor(src_lengths, device=device).contiguous()
        tgt_lengths_tensor = torch.tensor(tgt_lengths, device=device).contiguous()
        
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
                + x[:src_max_len]
                + ([] if self.src_eos_token is None else [self.src_eos_token])
                + ["<pad>"] * max(0, src_max_len - len(x)))
            src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(x)))
            
        tgt_padded, tgt_lengths = [], []
        for x in tgt_minibatch:
            tgt_padded.append(
                ([] if self.tgt_init_token is None else [self.tgt_init_token])
                + x[:tgt_max_len]
                + ([] if self.tgt_eos_token is None else [self.tgt_eos_token])
                + ["<pad>"] * max(0, tgt_max_len - len(x)))
            tgt_lengths.append(len(tgt_padded[-1]) - max(0, tgt_max_len - len(x)))
            
        return ((src_padded, src_lengths), (tgt_padded, tgt_lengths))
    
    @staticmethod
    def build_vocab(counter, max_vocab, init_token, eos_token):
        """Construct the Vocab object from dataset.

        Arguments:
            counter : collections.Counter object holding the frequencies of 
            each value found in the data.
            max_vocab : The maximum size of the vocabulary
            init_token : BOS token
            eos_token : EOS token
        """
                    
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
        num_parallel_reads : The number of processes to read text files at the same time.
    """
        
    def __init__(self, path, exts, max_length, num_parallel_reads):
        
        self.src_sentences = []
        self.trg_sentences = []
        self.sort_key = []
        self.max_length = max_length

        self.len_dataset = 0
        self.src_counters = Counter()
        self.trg_counters = Counter()

        src_path_list = sorted(Path('.').glob(path + '_' + exts[0] +'*'))
        trg_path_list = sorted(Path('.').glob(path + '_' + exts[1] +'*'))
        path_list = list(zip(src_path_list, trg_path_list))

        p = Pool(num_parallel_reads)
        result = p.map(self.read_files, path_list)
        p.close()
        p.join()

        for src_lines, trg_lines, key, src_counter, trg_counter in result:
            self.src_sentences += src_lines
            self.trg_sentences += trg_lines
            self.sort_key += key
            self.src_counters.update(src_counter)
            self.trg_counters.update(trg_counter)

        self.len_dataset = len(self.sort_key)
        print(f"len_dataset : {self.len_dataset}")

    def read_files(self, path):
        """Read text files, split them into sentences, and get counters for vocabulary. 
        """

        src_path, trg_path = path
        src_sentences = []
        trg_sentences = []
        sort_key = []

        src_counter = Counter()
        trg_counter = Counter()

        with src_path.open(encoding='utf-8') as src_file, trg_path.open(encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip().split(), trg_line.strip().split()
                src_length, trg_length = len(src_line), len(trg_line)
                if self.max_length and self.max_length < max(src_length, trg_length):
                    continue
                src_sentences.append(src_line)
                src_counter.update(src_line)
                trg_sentences.append(trg_line)
                trg_counter.update(trg_line)
                sort_key.append(trg_length + self.max_length * src_length)

        return src_sentences, trg_sentences, sort_key, src_counter, trg_counter

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
        num_parallel_reads : The number of processes to read text files at the same time.
    """

    def __init__(self, path, exts, max_length, num_parallel_reads):
        
        self.src_line_offset = []
        self.trg_line_offset = []
        self.file_offset_list = []
        self.sort_key = []
        self.max_length = max_length

        self.len_dataset = 0
        self.src_counters = Counter()
        self.trg_counters = Counter()

        self.src_path_list = sorted(Path('.').glob(path + '_' + exts[0] +'*'))
        self.trg_path_list = sorted(Path('.').glob(path + '_' + exts[1] +'*'))
        file_offset_list = list(range(len(self.src_path_list)))
        path_list = list(zip(self.src_path_list, self.trg_path_list, file_offset_list))

        p = Pool(num_parallel_reads)
        result = p.map(self.read_files, path_list)
        p.close()
        p.join()

        for src_offset, trg_offset, file_offset, key, src_counter, trg_counter in result:
            self.src_line_offset += src_offset
            self.trg_line_offset += trg_offset
            self.file_offset_list += file_offset
            self.sort_key += key
            self.src_counters.update(src_counter)
            self.trg_counters.update(trg_counter)

        self.len_dataset = len(self.sort_key)
        print(f"len_dataset : {self.len_dataset}")

    def read_files(self, path):
        """Read text files, calculate the offset of each sentence, and get counters for vocabulary. 
        """

        src_path, trg_path, file_offset = path
        src_line_offset = []
        trg_line_offset = []
        file_offset_list = []
        sort_key = []

        src_counter = Counter()
        trg_counter = Counter()

        with src_path.open(encoding='utf-8') as src_file, trg_path.open(encoding='utf-8') as trg_file:
            while True:
                src_offset, trg_offset = src_file.tell(), trg_file.tell()
                src_line, trg_line = src_file.readline(), trg_file.readline()
                src_line, trg_line = src_line.strip().split(), trg_line.strip().split()
                src_length, trg_length = len(src_line), len(trg_line)
                if self.max_length and self.max_length < max(src_length, trg_length):
                    continue
                if src_line and trg_line:
                    src_line_offset.append(src_offset)
                    src_counter.update(src_line)
                    trg_line_offset.append(trg_offset)
                    trg_counter.update(trg_line)
                    file_offset_list.append(file_offset)
                    sort_key.append(trg_length + self.max_length * src_length)
                else:
                    break

        return src_line_offset, trg_line_offset, file_offset_list, sort_key, src_counter, trg_counter
    
    def __getitem__(self, index):

        src_offset = self.src_line_offset[index]
        trg_offset = self.trg_line_offset[index]
        file_offset = self.file_offset_list[index]
        src_path = self.src_path_list[file_offset]
        trg_path = self.trg_path_list[file_offset]
        
        with src_path.open(encoding='utf-8') as src_file, trg_path.open(encoding='utf-8') as trg_file:
            src_file.seek(src_offset)
            trg_file.seek(trg_offset)
            src_sentence = src_file.readline().strip().split()
            trg_sentence = trg_file.readline().strip().split()

            return (src_sentence, trg_sentence)
        
    def __len__(self):
        return self.len_dataset

class BucketSampler(DistributedSampler):
    """Custom DistributedSampler for learning translators. 
    Sorting is performed to reduce the number of pads. However, if the entire sorting is performed, 
    the shuffle effect disappears, so sorting is performed only as much as the bucket size. 

    Arguments:
        bucket_size : Sorting is performed only as much as the bucket size. 
        The bucket_size must be divided by the batch_size.
        batch_size : The number of sentences in one batch.
        sort_key : Sort_key for sorting.
        **kwargs : Arguments of the inheriting DistributedSampler class. 
        (dataset, num_replicas, rank, shuffle(default = True), seed, drop_last)
    """

    def __init__(self, bucket_size, batch_size, sort_key, **kwargs):
    
        super(BucketSampler, self).__init__(**kwargs)

        if bucket_size % batch_size != 0:
            raise Except('The bucket_size must be divided by the batch_size.')
        self.bucket_size = bucket_size
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.num_buckets = ceil(self.num_samples / self.bucket_size)
    
    def __iter__(self):
    
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g1 = torch.Generator()
            g1.manual_seed(self.epoch * 3 + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g1).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        sub_indices_list = []
        g2 = torch.Generator()
        g2.manual_seed(self.epoch * 5 + self.seed)

        # To sort while maintaining the shuffle effect. 
        for bucket_index in range(self.num_buckets):
            start_index = self.bucket_size * bucket_index
            end_index = self.bucket_size * (bucket_index + 1)
            sub_indices = indices[start_index:end_index]
            sub_indices.sort(key = lambda idx : self.sort_key[idx])
            sub_indices_list += sub_indices

        if self.shuffle:
            new_indices = []
            num_extra = len(sub_indices_list) % self.batch_size
            extra_indices = sub_indices_list[-num_extra:]
            temp_indices = torch.randperm(len(sub_indices_list) // self.batch_size, 
                                          generator=g2).tolist()

            for temp_indice in temp_indices:
                new_indices += sub_indices_list[(temp_indice * self.batch_size):((temp_indice + 1) * self.batch_size)]
        else:
            new_indices = sub_indices_list
            
        return iter(new_indices)
        
class Batch():
    """The class representing an batch of a dataset. 
    It consists of source examples and target examples. 
    """
        
    def __init__(self, src_examples, tgt_examples):
        self.src = src_examples
        self.tgt = tgt_examples

def Dataloader_Test(train_fn:str=None,
                 valid_fn:str=None,
                 exts:Tuple[str, str]=None,
                 batch_size:int=64,
                 device:str='cpu',
                 max_vocab:int=99999999,
                 max_length:int=255,
                 fix_length:int=None,
                 use_bos:bool=True,
                 use_eos:bool=True,
                 shuffle:bool=True,
                 dsl:bool=False,
                 lazy:bool=True,
                 num_workers:int=0,
                 pin_memory:bool=False,
                 num_parallel_reads:int=2
                 ):
    """Test the dataloader. Outputs the vocabulary size and one batch of source and target. 
        Also, the two examples in the batch are converted into strings and displayed. 
    
    Arguments:
        train_fn : Training set file path(str) except the extention. (ex: train_en --> train)
        valid_fn : Validation set file path(str) except the extention. (ex: valid_en --> valid)
        exts : A tuple containing the extension to path for each language.
        batch_size : Mini batch size for gradient descent.
        device : The device containing the batch.
        max_vocab : The maximum size of the vocabulary, or None for no maximum. 
        max_length : The maximum number of words in a sentence. 
        fix_length: A fixed length that all examples will be padded to, 
        or None for flexible sequence lengths. 
        use_bos : Use the BOS token.  
        use_eos : Use the EOS token.  
        shuffle : Whether to shuffle examples between epochs. 
        dsl : Turn on dual-supervised learning mode. 
        lazy : If true, the dataloader reads the text file one line each time. 
        It prevents memory shortage caused by large datasets, but can cause I/O bottlenecks. 
        num_workers : how many subprocesses to use for data loading. 
        0 means that the data will be loaded in the main process.
        pin_memory : If True, the data loader will copy Tensors into CUDA pinned memory 
        before returning them. 
        num_parallel_reads : The number of processes to read text files at the same time.
        num_parallel_reads is the number of processes when the dataloader is first initialized, 
        and num_workers is the number of processes when loading data from the dataloader. 
    """
    
    config = locals()

    print("[Load the Dataloader]")
    start_time = time.time()
    memory_usage()

    loader = TranslationDataLoader(**config)

    execution_time = time.time() - start_time
    print(f"execution time: {execution_time: 10.5f} (seconds)")
    memory_usage()
    
    src_vocab_itos = loader.src_vocab.itos
    tgt_vocab_itos = loader.tgt_vocab.itos
    
    print(f'src_vocab_size : {len(src_vocab_itos)}')
    print(f'tgt_vocab_size : {len(tgt_vocab_itos)}')
    print('')

    print("[Generate Data]")
    start_time = time.time()
    memory_usage()

    for batch_index, batch in enumerate(loader.train_loader):
        for index in range(5):
            src_sentence_int = batch.src[0][index]
            src_sentence_str = ' '.join(list(map(lambda x : src_vocab_itos[x], src_sentence_int)))
            print(src_sentence_str)
            
            tgt_sentence_int = batch.tgt[0][index]
            tgt_sentence_str = ' '.join(list(map(lambda x : tgt_vocab_itos[x], tgt_sentence_int)))
            print(tgt_sentence_str)
            
        if batch_index > 100:
            break

    execution_time = time.time() - start_time
    print(f"execution time: {execution_time: 10.5f} (seconds)")
    memory_usage()

if __name__ == '__main__':
    fire.Fire({'test' : Dataloader_Test})
