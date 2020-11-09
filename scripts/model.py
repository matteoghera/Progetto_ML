from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_bert import BertEmbeddings
from torch import nn
import torch
import numpy as np


class MT_DNN(nn.Module):
    def __init__(self, config, max_len, task):
        # the BERT implementation of Pytorch run the lexicon and the trasformer encoder level
        super(MT_DNN, self).__init__()
        self.bert = BertModel.from_pretrained(config)
        self.max_len = max_len
        self.dropout = nn.Dropout(p=task.get_dropout_parameter())
        self.obj_function = task.get_objective_function(self.bert.config.hidden_size)  ### not implemented

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, encoder_hidden_states):
        embeddings = BertEmbeddings(self.bert.config)
        embedding_output = embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=None
        )
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states
        )
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.obj_function(last_hidden_state, pooled_output, embedding_output)
        return last_hidden_state, pooled_output


class ModelManager:
    def __init__(self, task, config, max_len, device, epochs):
        self.task = task

        model = MT_DNN(config, max_len, task)
        self.model = model.to(device)

        _, dev_tokenized_data_loader = task.get_dev()

        self.optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
        total_steps = len(dev_tokenized_data_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps
        )
        self.loss_fn = task.get_loss_function().to(device)  ### not implemented


class ModellingHelper:
    def __init__(self, tasks, config, max_len):
        self.config = config
        self.max_len = max_len
        self.tasks = tasks  ### tasks=[cola, sst_2, mnli, rte, wnli, qqp, mrpc, snli, sts_b, qnli]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epochs):
        self.model_manager_list = []
        for task in self.tasks:
            self.model_manager_list.append(ModelManager(task, self.config, self.max_len, self.device, epochs))

        encoder_hidden_states = None
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            for model_manager in self.model_manager_list:
                current_task=model_manager.task
                print(current_task.get_name())

                dev, dev_tokenized_data_loader = current_task.get_dev()
                encoder_hidden_states = self.__train_epoch(model_manager.model, dev_tokenized_data_loader,
                                                           model_manager.loss_fn, model_manager.optimizer,
                                                           model_manager.scheduler, encoder_hidden_states,
                                                           len(dev))

                train_loss = self.train_loss[len(self.train_loss) - 1]
                train_acc = self.train_acc[len(self.train_acc) - 1]
                print(f'Train loss {train_loss} accuracy {train_acc}')

                val, val_tokenized_data_loader=current_task.get_dev()
                val_acc, val_loss=self.__eval_model(model_manager.model, val_tokenized_data_loader, model_manager.loss_fn, encoder_hidden_states, len(val))

                self.val_acc.append(val_acc)
                self.val_loss.append(val_loss)
                print(f'Train loss {val_loss} accuracy {val_acc}')
                print()
        self.last_hidden_state=encoder_hidden_states

    def __train_epoch(self,
                      model,
                      data_loader,  # train data loader
                      loss_fn,
                      optimizer,
                      scheduler,
                      encoder_hidden_states,
                      n_examples
                      ):

        model = model.train()

        losses = []
        correct_predictions = 0

        for d in data_loader:
            input_ids = d['input_ids'].to(self.device)
            attention_mask = d['attention_mask'].to(self.device)
            token_type_ids = d["token_type_ids"].to(self.device)
            position_ids = d['positional_encoding'].to(self.device)
            targets = d['targets'].to(self.device)

            encoder_hidden_states, pooled_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states
            )

            _, preds = torch.max(pooled_output, dim=1)
            loss = loss_fn(pooled_output, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        self.train_acc.append(correct_predictions.double() / n_examples)
        self.train_loss.append(np.mean(losses))
        return encoder_hidden_states  # correct_predictions.double() / n_examples, np.mean(losses),

    def __eval_model(self,
                     model,
                     data_loader,  # validation data loader
                     loss_fn,
                     encoder_hidden_states,
                     n_examples):

        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d['input_ids'].to(self.device)
                attention_mask = d['attention_mask'].to(self.device)
                token_type_ids = d["token_type_ids"].to(self.device)
                position_ids = d['positional_encoding'].to(self.device)
                targets = d['targets'].to(self.device)

                encoder_hidden_states, pooled_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    encoder_hidden_states=encoder_hidden_states
                )

                _, preds = torch.max(pooled_output, dim=1)
                loss = loss_fn(pooled_output, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())
        return correct_predictions.double()/n_examples, np.mean(losses)
