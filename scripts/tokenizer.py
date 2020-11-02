from transformers import BertTokenizer
from torch.utils import data

import torch

class DatasetPlus:
    def __init__(self, df,  tokenizer, max_len, batch_size, num_workers, column_sequence1, column_sequence2=None, column_target=None, dtype=torch.long):
        if not column_sequence2 is None and not column_target is None:
            ds = self.DatasetRow(
                sequence1=df[column_sequence1].to_numpy(),
                sequence2=df[column_sequence2].to_numpy(),
                targets=df[column_target].to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len,
                dtype = dtype

            )
        elif column_sequence2 is None and not column_target is None:
            ds = self.DatasetRow(
                sequence1=df[column_sequence1].to_numpy(),
                targets=df[column_target].to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len,
                dtype = dtype
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





    class DatasetRow(data.Dataset):
        def __init__(self, tokenizer, max_len, sequence1, sequence2=None, targets=None, dtype=torch.long):
            self.sequence1 = sequence1
            self.sequence2 = sequence2
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.dtype=dtype

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
                    'token_type_ids':encoding['token_type_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
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
                    'attention_mask': encoding['attention_mask'].flatten(),
                }