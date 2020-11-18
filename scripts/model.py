from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import numpy as np


class MT_DNN(nn.Module):
    def __init__(self, config, max_len):
        # the BERT implementation of Pytorch run the lexicon and the trasformer encoder level
        super(MT_DNN, self).__init__()
        self.bert = BertModel.from_pretrained(config)  ## only one BERT model
        self.max_len = max_len
        self.obj_function=None

    def forward(self, task, input_ids, attention_mask, token_type_ids):
        dropout = nn.Dropout(p=task.get_dropout_parameter())
        self.obj_function = task.get_objective_function(self.bert.config.hidden_size)

        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = dropout(pooled_output)
        pooled_output = self.obj_function(pooled_output)
        return last_hidden_state, pooled_output


class ModelManager:
    def __init__(self, task, model, device, epochs):
        self.task = task

        _, dev_tokenized_data_loader = task.get_dev()

        self.optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
        total_steps = len(dev_tokenized_data_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps
        )
        self.loss_fn = task.get_loss_function().to(device)

        self.train_acc = {}
        self.train_loss = {}
        self.val_acc = {}
        self.val_loss = {}

    def save_train_metrics(self,epoch, loss, acc, phase):
        if phase.__eq__("train"):
            self.train_loss.update({epoch:loss})
            self.train_acc.update({epoch: acc})
        else:
            self.val_loss.update({epoch:loss})
            self.val_acc.update({epoch: acc})
        self.task.print_metrics(loss, acc, phase) ## not implemented



class ModellingHelper:
    def __init__(self, tasks, config, max_len):
        self.config = config
        self.tasks = tasks  ### tasks=[cola, sst_2, mnli, rte, wnli, qqp, mrpc, snli, sts_b, qnli]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MT_DNN(config, max_len)
        self.model = self.model.to(self.device)

    def train(self, epochs):
        self.model_manager_list = []
        for task in self.tasks:
            self.model_manager_list.append(ModelManager(task, self.model, self.device, epochs))


        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            for model_manager in self.model_manager_list:
                current_task=model_manager.task
                print(current_task.get_name())

                dev, dev_tokenized_data_loader = current_task.get_dev()
                train_acc, train_loss = self.__train_epoch(current_task,dev_tokenized_data_loader,
                                                           model_manager.loss_fn, model_manager.optimizer,
                                                           model_manager.scheduler, len(dev))
                model_manager.save_train_metrics(epoch, train_loss, train_acc, "train")
                del dev, dev_tokenized_data_loader, train_acc, train_loss

                val, val_tokenized_data_loader=current_task.get_dev()
                val_acc, val_loss=self.__eval_model(current_task, val_tokenized_data_loader, model_manager.loss_fn, len(val))
                model_manager.save_train_metrics(epoch, val_loss, val_acc, "eval")
                del val, val_tokenized_data_loader, train_acc, train_loss

    def __train_epoch(self,
                      current_task,
                      data_loader,  # train data loader
                      loss_fn,
                      optimizer,
                      scheduler,
                      n_examples):

        self.model = self.model.train()

        losses = []
        accs=[]
        for d in data_loader:
            input_ids = d['input_ids'].to(self.device)
            attention_mask = d['attention_mask'].to(self.device)
            token_type_ids = d["token_type_ids"].to(self.device)
            targets = d['targets'].to(self.device)

            encoder_hidden_states, pooled_output = self.model(
                task=current_task,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            _, preds = torch.max(pooled_output, dim=1)
            loss = loss_fn(pooled_output, targets)
            losses.append(loss.item())

            accs.append(current_task.compute_matric_value(preds, targets, n_examples))

            loss.backward()
            del input_ids, attention_mask, token_type_ids, targets, encoder_hidden_states, pooled_output, preds, loss
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return np.mean(accs),np.mean(losses)

    def __eval_model(self,
                     current_task,
                     data_loader,  # validation data loader
                     loss_fn,
                     n_examples
                     ):

        self.model = self.model.eval()

        losses = []
        accs=[]
        with torch.no_grad():
            for d in data_loader:
                input_ids = d['input_ids'].to(self.device)
                attention_mask = d['attention_mask'].to(self.device)
                token_type_ids = d["token_type_ids"].to(self.device)
                targets = d['targets'].to(self.device)

                encoder_hidden_states, pooled_output = self.model(
                    task=current_task,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                _, preds = torch.max(pooled_output, dim=1)
                loss = loss_fn(pooled_output, targets)

                losses.append(loss.item())
                accs.append(current_task.compute_matric_value(preds, targets, n_examples))  ##not implemented
        return np.mean(accs), np.mean(losses)
