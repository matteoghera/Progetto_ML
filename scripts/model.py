from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import numpy as np
from pandas import DataFrame


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MT_DNN(nn.Module):
    def __init__(self, config, max_len, fine_tuning=True):
        # the BERT implementation of Pytorch run the lexicon and the trasformer encoder level
        super(MT_DNN, self).__init__()
        self.bert = BertModel.from_pretrained(config)  ## only one BERT model
        self.config=self.bert.config
        if not fine_tuning:
            self.bert=BertModel(self.config)
        self.max_len = max_len

    def forward(self,input_ids, attention_mask, token_type_ids, p, obj_function):
        dropout = nn.Dropout(p=p).to(device)
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = dropout(pooled_output)
        pooled_output = obj_function(pooled_output)
        return last_hidden_state, pooled_output


class ModelManager:
    def __init__(self, task, model, epochs):
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

        self.train_results = []
        self.val_results = []

    def save_train_metrics(self,epoch, loss, acc, phase):
        if phase.__eq__("train"):
            self.train_results.append({"name": self.task.get_name(), "epoch": epoch, "loss":loss, "acc":acc})
        else:
            self.val_results.append({"name": self.task.get_name(), "epoch": epoch, "loss":loss, "acc":acc})
        self.task.print_metrics(loss, acc, phase)



class ModellingHelper:
    def __init__(self, tasks, config, max_len, models_path, fine_tuning=True):
        self.tasks = tasks  ### tasks=[cola, sst_2, mnli, rte, wnli, qqp, mrpc, snli, sts_b, qnli]
        self.models_path=models_path

        model = MT_DNN(config, max_len, fine_tuning=fine_tuning)
        self.model = model.to(device)


    def train(self, epochs):
        self.model_manager_list = []
        for task in self.tasks:
            self.model_manager_list.append(ModelManager(task, self.model, epochs))

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            print("\tTRAINING PHASE")
            for model_manager in self.model_manager_list:
                current_task=model_manager.task
                print(current_task.get_name())

                dev, dev_tokenized_data_loader = current_task.get_dev()
                train_acc, train_loss = self.__train_epoch(current_task,dev_tokenized_data_loader,
                                                           model_manager.loss_fn, model_manager.optimizer,
                                                           model_manager.scheduler, len(dev))
                model_manager.save_train_metrics(epoch, train_loss, train_acc, "train")
                del dev, dev_tokenized_data_loader, train_acc, train_loss


            print("\tVALIDATION PHASE")
            best_accuracy=0
            for model_manager in self.model_manager_list:
                current_task = model_manager.task
                print(current_task.get_name())

                val, val_tokenized_data_loader=current_task.get_train()
                val_acc, val_loss=self.__eval_model(current_task, val_tokenized_data_loader,
                                                    model_manager.loss_fn, len(val))
                model_manager.save_train_metrics(epoch, val_loss, val_acc, "eval")

                if val_acc > best_accuracy:  # we save the model with the best accuracy
                    torch.save(self.model.state_dict(), self.models_path/'best_MT_DNN.bin')
                    best_accuracy = val_acc

                del val, val_tokenized_data_loader, val_acc, val_loss

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
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets = d['targets'].to(device)

            obj_function=current_task.get_objective_function(self.model.bert.config.hidden_size).to(device)
            p=current_task.get_dropout_parameter()
            encoder_hidden_states, pooled_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                obj_function=obj_function,
                p=p
            )
            preds=current_task.predict(pooled_output)
            loss = loss_fn(pooled_output, targets)
            losses.append(loss.item())

            print(f'Pred: {preds} \t targets: {targets}')
            accs.append(current_task.compute_matric_value(preds.cpu().data.numpy(), targets.cpu().data.numpy(), n_examples))

            loss.backward()
            del input_ids, attention_mask, token_type_ids, targets, encoder_hidden_states, pooled_output, preds, loss, obj_function, p
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
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                token_type_ids = d["token_type_ids"].to(device)
                targets = d['targets'].to(device)

                obj_function = current_task.get_objective_function(self.model.config.hidden_size).to(device)
                p = current_task.get_dropout_parameter()
                encoder_hidden_states, pooled_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    obj_function=obj_function,
                    p=p
                )
                preds=current_task.predict(pooled_output)
                loss = loss_fn(pooled_output, targets)
                losses.append(loss.item())
                accs.append(current_task.compute_matric_value(preds.cpu().data.numpy(), targets.cpu().data.numpy(), n_examples))
                del input_ids, attention_mask, token_type_ids, targets, encoder_hidden_states, pooled_output, preds, loss, obj_function, p
        return np.mean(accs), np.mean(losses)

    def view_result(self):
        training_results=[]
        evaluation_results=[]
        for model_manager in self.model_manager_list:
            training_results.extend(model_manager.train_results)
            evaluation_results.extend(model_manager.val_results)
        return DataFrame(training_results), DataFrame(evaluation_results)
