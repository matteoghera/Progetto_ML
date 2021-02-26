from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import numpy as np
from pandas import DataFrame
from time import sleep


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerEncoder(nn.Module):
    def __init__(self, config, max_len, fine_tuning=True):
        # the BERT implementation of Pytorch run the lexicon and the trasformer encoder level
        super(TransformerEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config)  ## only one BERT model
        self.config=self.bert.config
        if not fine_tuning:
            self.bert=BertModel(self.config)
        self.hidden_size=self.bert.config.hidden_size
        self.max_len = max_len

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output=last_hidden_state[:,0]
        return last_hidden_state, pooled_output


class MT_DNN(nn.Module):
    def __init__(self, encoder, obj_function, p):
        super(MT_DNN, self).__init__()
        self.encoder=encoder
        self.obj_function=obj_function
        self.dropout=nn.Dropout(p=p)

    def forward(self,input_ids, attention_mask, token_type_ids):
        last_hidden_state, pooled_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.obj_function(last_hidden_state, pooled_output)
        return pooled_output


class ModelManager:
    def __init__(self, task, encoder, epochs):
        self.task = task

        self.encoder=encoder
        model=MT_DNN(self.encoder, self.task.get_objective_function(encoder.hidden_size), self.task.get_dropout_parameter())
        self.model=model.to(device)

        #self.optimizer = AdamW(self.model.parameters(), lr=5e-5, correct_bias=False)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=5e-5)
        total_steps = task.TOTAL_BATCH * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps
        )
        self.loss_fn = task.get_loss_function().to(device)

        self.train_results = []
        self.val_results = []
        self.best_accuracy=-2


    def save_model(self, val_acc, models_path):
        if val_acc > self.best_accuracy:
            torch.save(self.encoder.state_dict(), models_path / self.task.get_name() + '_encoder.bin')
            torch.save(self.model.state_dict(), models_path / self.task.get_name()+'_MT_DNN.bin')
            self.best_accuracy = val_acc

    def save_train_metrics(self,epoch, loss, acc, phase):
        if phase.__eq__("train"):
            self.train_results.append({"name": self.task.get_name(), "epoch": epoch, "loss":loss, "metric_value":acc})
        else:
            self.val_results.append({"name": self.task.get_name(), "epoch": epoch, "loss":loss, "metric_value":acc})
        self.task.print_metrics(loss, acc, phase)



class ModellingHelper:
    def __init__(self, tasks, config, max_len, models_path, fine_tuning=True):
        self.tasks = tasks  ### tasks=[cola, sst_2, mnli, rte, wnli, qqp, mrpc, snli, sts_b, qnli]
        self.models_path=models_path

        encoder = TransformerEncoder(config, max_len, fine_tuning=fine_tuning)
        self.encoder = encoder.to(device)


    def train(self, epochs):
        self.model_manager_list = []
        for task in self.tasks:
            self.model_manager_list.append(ModelManager(task, self.encoder, epochs))

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            print("\tTRAINING PHASE")
            for model_manager in self.model_manager_list:
                current_task=model_manager.task
                print(current_task.get_name())
                i=0

                train, train_tokenized_data_loader = current_task.get_train()
                #train, train_tokenized_data_loader = current_task.get_dev()
                train_acc, train_loss, i = self.__train_epoch(model_manager.model,current_task, train_tokenized_data_loader,
                                                           model_manager.loss_fn, model_manager.optimizer,
                                                           model_manager.scheduler, len(train), i)
                print()
                model_manager.save_train_metrics(epoch+1, train_loss, train_acc, "train")
                del train, train_tokenized_data_loader, train_acc, train_loss, i


            print("\tVALIDATION PHASE")
            for model_manager in self.model_manager_list:
                current_task = model_manager.task
                print(current_task.get_name())
                i=0

                val, val_tokenized_data_loader=current_task.get_dev()
                val_acc, val_loss, i=self.__eval_model(model_manager.model, current_task, val_tokenized_data_loader,
                                                    model_manager.loss_fn, len(val), i)
                print()
                model_manager.save_train_metrics(epoch+1, val_loss, val_acc, "eval")
                model_manager.save_model(val_acc, self.models_path)
                del val, val_tokenized_data_loader, val_acc, val_loss
            print()

    def __train_epoch(self,
                      model,
                      current_task,
                      data_loader,  # train data loader
                      loss_fn,
                      optimizer,
                      scheduler,
                      n_examples,
                      index):

        model=model.train()

        #n_batches=round(n_examples/data_loader.batch_size)
        n_batches = len(data_loader)
        losses = []
        accs=[]
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets = d['targets'].to(device)

            pooled_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            preds=current_task.predict(pooled_output)
            loss = loss_fn(pooled_output, targets)
            losses.append(loss.item())
            index += 1
            print(f'\rBatch ID: {index} / {n_batches} \t Pred: {np.around(preds.cpu().data.numpy(), decimals=1)} \t Targets: {np.around(targets.cpu().data.numpy(), decimals=1)}', end=" ")
            accs.append(current_task.compute_matric_value(preds.cpu().data.numpy(), targets.cpu().data.numpy(), n_examples))

            loss.backward()
            del input_ids, attention_mask, token_type_ids, targets, pooled_output, preds, loss
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            #sleep(1)
        return np.mean(accs),np.mean(losses), index

    def __eval_model(self,
                     model,
                     current_task,
                     data_loader,  # validation data loader
                     loss_fn,
                     n_examples,
                     index
                     ):

        model=model.eval()

        #n_batches = round(n_examples / data_loader.batch_size)
        n_batches = len(data_loader)
        losses = []
        accs=[]
        with torch.no_grad():
            for d in data_loader:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                token_type_ids = d["token_type_ids"].to(device)
                targets = d['targets'].to(device)

                pooled_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                preds=current_task.predict(pooled_output)
                loss = loss_fn(pooled_output, targets)
                losses.append(loss.item())
                index+=1
                print(f'\rBatch ID: {index} / {n_batches} \t Pred: {np.around(preds.cpu().data.numpy(), decimals=1)} \t Targets: {np.around(targets.cpu().data.numpy(), decimals=1)}', end=" ")
                accs.append(current_task.compute_matric_value(preds.cpu().data.numpy(), targets.cpu().data.numpy(), n_examples))
                del input_ids, attention_mask, token_type_ids, targets, pooled_output, preds, loss
                #sleep(1)
        return np.mean(accs), np.mean(losses), index

    def view_result(self):
        training_results=[]
        evaluation_results=[]
        for model_manager in self.model_manager_list:
            training_results.extend(model_manager.train_results)
            evaluation_results.extend(model_manager.val_results)
        return DataFrame(training_results), DataFrame(evaluation_results)
