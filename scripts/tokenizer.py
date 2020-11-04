from transformers import BertTokenizer
from torch.utils import data
from math import sin, cos

import torch


class DatasetPlus:
    def __init__(self, df, tokenizer, max_len, batch_size, num_workers, column_sequence1, column_sequence2=None,
                 column_target=None, dtype=torch.long):
        if not column_sequence2 is None and not column_target is None:
            ds = self.DatasetRow(
                sequence1=df[column_sequence1].to_numpy(),
                sequence2=df[column_sequence2].to_numpy(),
                targets=df[column_target].to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len,
                dtype=dtype

            )
        elif column_sequence2 is None and not column_target is None:
            ds = self.DatasetRow(
                sequence1=df[column_sequence1].to_numpy(),
                targets=df[column_target].to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len,
                dtype=dtype
            )
        else:
            ds = self.DatasetRow(
                sequence1=df[column_sequence1].to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len
            )
        self.my_dataloader = data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def viewData(self):
        return next(self.my_dataloader.__iter__())

    def get_dataloader(self):
        return self.my_dataloader

    class DatasetRow(data.Dataset):
        def __init__(self, tokenizer, max_len, sequence1, sequence2=None, targets=None, dtype=torch.long):
            self.sequence1 = sequence1
            self.sequence2 = sequence2
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.dtype = dtype

        def __len__(self):
            return len(self.sequence1)

        def __getitem__(self, item):
            # item enumerates from 0 to BATCH_SIZE
            if not self.sequence2 is None and not self.targets is None:
                sequence1 = str(self.sequence1[item])
                sequence2 = str(self.sequence2[item])
                target = self.targets[item]
                encoding = self.tokenizer.encode_plus(
                    text=sequence1,
                    text_pair=sequence2,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                return {
                    'sequence1': sequence1,
                    'sequence2': sequence2,
                    'input_ids': encoding['input_ids'].flatten(),
                    'token_type_ids': encoding['token_type_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'positional_encoding': torch.tensor(
                        self.positional_encoding(encoding['input_ids'].flatten().tolist()), dtype=torch.double),
                    'targets': torch.tensor(target, dtype=self.dtype)
                }
            elif self.sequence2 is None and not self.targets is None:
                sequence1 = str(self.sequence1[item])
                target = self.targets[item]
                encoding = self.tokenizer.encode_plus(
                    text=sequence1,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                return {
                    'sequence1': sequence1,
                    'input_ids': encoding['input_ids'].flatten(),
                    'token_type_ids': encoding['token_type_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'positional_encoding': torch.tensor(
                        self.positional_encoding(encoding['input_ids'].flatten().tolist()), dtype=torch.double),
                    'targets': torch.tensor(target, dtype=self.dtype)
                }
            else:
                sequence1 = str(self.sequence1[item])
                encoding = self.tokenizer.encode_plus(
                    text=sequence1,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                return {
                    'sequence1': sequence1,
                    'input_ids': encoding['input_ids'].flatten(),
                    'token_type_ids': encoding['token_type_ids'].flatten(),
                    'positional_encoding': torch.tensor(
                        self.positional_encoding(encoding['input_ids'].flatten().tolist()), dtype=torch.double),
                    'attention_mask': encoding['attention_mask'].flatten(),
                }

        def positional_encoding(self, input_ids):
            d_model = self.max_len

            first_sep_token_id = 0
            find = False
            while not find:
                first_sep_token_id += 1
                if input_ids[first_sep_token_id] == 102:
                    find = True

            num_tokens_seq_1 = len(input_ids[:first_sep_token_id + 1])
            num_tokens_seq_2 = d_model - num_tokens_seq_1
            positional_encoding = []
            for pos in range(d_model):
                if pos < num_tokens_seq_1:
                    if num_tokens_seq_1 % 2 == 0:
                        positional_encoding.append(sin(pos / (10000 ** ((2 * num_tokens_seq_1) / d_model))))
                    else:
                        positional_encoding.append(cos(pos / (10000 ** ((2 * num_tokens_seq_1) / d_model))))
                else:
                    if num_tokens_seq_2 % 2 == 0:
                        positional_encoding.append(sin(pos / (10000 ** ((2 * num_tokens_seq_2) / d_model))))
                    else:
                        positional_encoding.append(cos(pos / (10000 ** ((2 * num_tokens_seq_2) / d_model))))
            return positional_encoding
