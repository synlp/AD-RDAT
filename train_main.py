
from pytorch_transformers import BertTokenizer,BertModel

from advMultiCri import AdvMultiCriModel

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np

from sklearn import metrics
import torch

from data_utils import DataProcessor
from diacritization_stat import calculate_der, calculate_wer

from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler
# from apex import amp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

BUCKWALTER_MAP = {
    '\'': '\u0621',
    '|': '\u0622',
    '>': '\u0623',
    'O': '\u0623',
    '&': '\u0624',
    'W': '\u0624',
    '<': '\u0625',
    'I': '\u0625',
    '}': '\u0626',
    'A': '\u0627',
    'b': '\u0628',
    'p': '\u0629',
    't': '\u062A',
    'v': '\u062B',
    'j': '\u062C',
    'H': '\u062D',
    'x': '\u062E',
    'd': '\u062F',
    '*': '\u0630',
    'r': '\u0631',
    'z': '\u0632',
    's': '\u0633',
    '$': '\u0634',
    'S': '\u0635',
    'D': '\u0636',
    'T': '\u0637',
    'Z': '\u0638',
    'E': '\u0639',
    'g': '\u063A',
    '_': '\u0640',
    'f': '\u0641',
    'q': '\u0642',
    'k': '\u0643',
    'l': '\u0644',
    'm': '\u0645',
    'n': '\u0646',
    'h': '\u0647',
    'w': '\u0648',
    'Y': '\u0649',
    'y': '\u064A',
    'F': '\u064B',
    'N': '\u064C',
    'K': '\u064D',
    'a': '\u064E',
    'u': '\u064F',
    'i': '\u0650',
    '~': '\u0651',
    'o': '\u0652',
    '`': '\u0670',
    '{': '\u0671',
}
ARABIC_LETTERS_LIST = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي'
CLASSES_LIST = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'َّ', 'ًّ', 'ُّ', 'ٌّ', 'ِّ', 'ٍّ']

def get_rank():
    import torch.distributed as dist
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

class Instructor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_processor = DataProcessor(args.data_dir, args.dataset)
        self.style = "Fadel"
        if args.dataset == "ATB":
            self.style = "Zitouni"
        elif args.dataset == "Tashkeela":
            self.style = "Fadel"

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))

    def saving_model(self, saving_model_path, model, optimizer):
        if not os.path.exists(saving_model_path):
            os.mkdir(saving_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(saving_model_path, CONFIG_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(model_to_save.config.to_json_string())
        torch.save({'optimizer': optimizer.state_dict(),
                    'master params': optimizer},
                   output_optimizer_file)
        output_args_file = os.path.join(saving_model_path, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def load_model(self, model, optimizer, saving_model_path):
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        #model
        checkpoint_model = torch.load(output_model_file, map_location="cpu")
        model.load_state_dict(checkpoint_model)
        #optimizer
        checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")

        optimizer.load_state_dict(checkpoint_optimizer["optimizer"])
        return model, optimizer

    def save_args(self):
        output_args_file = os.path.join(self.args.outdir, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def _train_multi_criteria(self, model, optimizer, scheduler, train_data_loader, farasa_data_loader, global_step, args):
        n_correct, n_total, loss_total = 0, 0, 0
        tr_loss = 0
        average_loss = 0

        # switch model to training mode
        model.train()
        for i_batch, (sample_batched_a, sample_batched_b) in enumerate(zip(train_data_loader, farasa_data_loader)):
            loss, d_loss, h_loss = 0, 0, 0
            for criteria_index, sample_batched in enumerate([sample_batched_a, sample_batched_b]):
                input_ids = sample_batched["input_ids"].to(self.args.device)
                segment_ids = sample_batched["segment_ids"].to(self.args.device)
                label_ids = sample_batched["label_ids"].to(self.args.device)
                valid_ids = sample_batched["valid_ids"].to(self.args.device)
                wordpiece_ids = sample_batched["wordpiece_ids"].to(self.args.device)
                c2w_map = sample_batched["c2w_map"].to(self.args.device)
                tag_seq, _loss, _d_loss, _h_loss = model(input_ids, segment_ids, valid_ids, label_ids,
                                                            wordpiece_ids=wordpiece_ids, c2w_map=c2w_map,
                                                         criteria_index=criteria_index)

                loss += _loss
                d_loss += _d_loss
                h_loss += _h_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                d_loss = d_loss / args.gradient_accumulation_steps
                h_loss = h_loss / args.gradient_accumulation_steps
            loss = loss / 2
            d_loss = d_loss / 2
            h_loss = h_loss / 2
            if args.adversary is True:
                scaled_loss = loss + d_loss + h_loss
                scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            average_loss += loss

            if (i_batch + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            n_correct += (tag_seq == label_ids).sum().item()
            n_total += len(tag_seq)
            loss_total += loss.item() * len(tag_seq)
            if global_step % self.args.log_step == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('global_step: {}, loss: {:.4f}, acc: {:.4f} '
                            'lr: {:.6f}'.format(global_step, train_loss, train_acc, optimizer.get_lr()[0]))
        return global_step

    def _train_single_criteria(self, model, optimizer, scheduler, train_data_loader, global_step, args):
        n_correct, n_total, loss_total = 0, 0, 0
        tr_loss = 0
        average_loss = 0

        # switch model to training mode
        model.train()
        for i_batch, sample_batched in enumerate(train_data_loader):
            input_ids = sample_batched["input_ids"].to(self.args.device)
            segment_ids = sample_batched["segment_ids"].to(self.args.device)
            label_ids = sample_batched["label_ids"].to(self.args.device)
            valid_ids = sample_batched["valid_ids"].to(self.args.device)
            wordpiece_ids = sample_batched["wordpiece_ids"].to(self.args.device)
            c2w_map = sample_batched["c2w_map"].to(self.args.device)
            tag_seq, loss, _d_loss, _h_loss = model(input_ids, segment_ids, valid_ids, label_ids,
                                                     wordpiece_ids=wordpiece_ids, c2w_map=c2w_map,
                                                     criteria_index=0)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            average_loss += loss
            if (i_batch + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            n_correct += (tag_seq == label_ids).sum().item()
            n_total += len(tag_seq)
            loss_total += loss.item() * len(tag_seq)
            if global_step % self.args.log_step == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('global_step: {}, loss: {:.4f}, acc: {:.4f} '
                            'lr: {:.6f}'.format(global_step, train_loss, train_acc, optimizer.get_lr()[0]))
        return global_step

    def _train(self, model, optimizer, scheduler, train_data_loader, farasa_data_loader, dev_dataloader):
        path = None

        global_step = 0
        num_of_no_improvement = 0

        args = self.args
        for epoch in range(self.args.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))

            if epoch < self.args.num_epoch_multi_cri and num_of_no_improvement < args.patient and args.multi_criteria is True:
                global_step = self._train_multi_criteria(model, optimizer, scheduler, train_data_loader, farasa_data_loader, global_step, args)
            else:
                global_step = self._train_single_criteria(model, optimizer, scheduler, train_data_loader, global_step, args)

            dev_result = self._evaluate_acc_f1(model, dev_dataloader)
            logger.info(dev_result)

        return path

    def _evaluate_acc_f1(self, model, data_loader, saving_path=None):
        n_correct, n_total, loss_total = 0, 0, 0
        label_map = self.data_processor.get_id2label_map()

        # switch model to evaluation mode
        model.eval()

        saving_path_f = open(saving_path, 'w') if saving_path is not None and is_main_process() else None

        result = {}
        predict_lines = []
        target_lines = []
        sample_idx = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                input_ids = sample_batched["input_ids"].to(self.args.device)
                segment_ids = sample_batched["segment_ids"].to(self.args.device)
                label_ids = sample_batched["label_ids"].to(self.args.device)
                valid_ids = sample_batched["valid_ids"].to(self.args.device)
                wordpiece_ids = sample_batched["wordpiece_ids"].to(self.args.device)
                c2w_map = sample_batched["c2w_map"].to(self.args.device)
                word_bound = sample_batched["word_bound"]
                word_list = sample_batched["word_list"]
                tag_seq, loss, d_loss, h_loss = model(input_ids, segment_ids, valid_ids, label_ids,
                                                      wordpiece_ids=wordpiece_ids, c2w_map=c2w_map)

                word_list = [w.split("\t") for w in word_list]

                n_correct += (tag_seq == label_ids).sum().item()
                n_total += len(tag_seq)
                loss_total += loss.item() * len(tag_seq)

                logits = tag_seq.to('cpu').numpy()
                label_ids = label_ids.to('cpu').numpy()

                for i, label in enumerate(label_ids):
                    predict_lines.append("")
                    target_lines.append("")
                    label_list = []
                    for j, m in enumerate(label):
                        gold = label_map[label_ids[i][j]]
                        label_list.append(gold)
                        if gold in ['[SEP]']:
                            break

                    for j, m in enumerate(label):
                        gold = label_map[label_ids[i][j]]
                        if gold in ['[CLS]']:
                            continue
                        if gold in ['[SEP]']:
                            break
                        pred = label_map[logits[i][j]]
                        if pred == "o":
                            pred = "#"
                        if gold == "o":
                            gold = "#"

                        input_token = word_list[i][j]
                        pred_token = BUCKWALTER_MAP.get(pred, "")
                        gold_token = BUCKWALTER_MAP.get(gold, "")
                        if input_token in list(ARABIC_LETTERS_LIST):
                            if pred_token in CLASSES_LIST:
                                predict_lines[-1] += input_token + pred_token
                            else:
                                predict_lines[-1] += input_token
                            if gold_token in CLASSES_LIST:
                                target_lines[-1] += input_token + gold_token
                            else:
                                target_lines[-1] += input_token
                        else:
                            predict_lines[-1] += input_token
                            target_lines[-1] += input_token

                        if word_bound[i][j] == 1:
                            predict_lines[-1] += " "
                            target_lines[-1] += " "

                    sample_idx += 1

        result["acc"] = n_correct / n_total
        result["loss"] = loss_total / n_total

        der = calculate_der(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style)
        wer = calculate_wer(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style)
        result["der"] = der
        result["wer"] = wer
        eval_type = "with_case_ending_including_no_diacritic"
        result["{}_der".format(eval_type)] = der
        result["{}_wer".format(eval_type)] = wer
        der = calculate_der(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style, case_ending=False)
        wer = calculate_wer(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style, case_ending=False)
        eval_type = "without_case_ending_including_no_diacritic"
        result["{}_der".format(eval_type)] = der
        result["{}_wer".format(eval_type)] = wer
        der = calculate_der(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style, no_diacritic=False)
        wer = calculate_wer(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style, no_diacritic=False)
        eval_type = "with_case_ending_excluding_no_diacritic"
        result["{}_der".format(eval_type)] = der
        result["{}_wer".format(eval_type)] = wer
        der = calculate_der(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style, case_ending=False, no_diacritic=False)
        wer = calculate_wer(predict_lines, target_lines, ARABIC_LETTERS_LIST, CLASSES_LIST, self.style, case_ending=False, no_diacritic=False)
        eval_type = "without_case_ending_excluding_no_diacritic"
        result["{}_der".format(eval_type)] = der
        result["{}_wer".format(eval_type)] = wer

        return result

    def prepare_model_optimizer(self):
        tag_size = self.data_processor.get_tag_size()
        model = AdvMultiCriModel(tag_size=tag_size, embedding=self.args.embedding, bert_path=self.args.bert_model,
                                 encoder=self.args.encoder, num_layers=self.args.num_layers)
        print("build model...")

        if self.args.resume is True:
            print("resume from: {}".format(self.args.resume_model))
            output_model_file = os.path.join(self.args.resume_model, WEIGHTS_NAME)
            checkpoint_model = torch.load(output_model_file, map_location="cpu")
            model.load_state_dict(checkpoint_model)
        else:
            model._reset_params(self.args.initializer)

        model = model.to(self.args.device)

        train_data_loader, farasa_data_loader, dev_dataloader = \
            self.data_processor.get_all_dataloader(self.tokenizer, self.args)

        num_train_optimization_steps = int(
            len(train_data_loader) / self.args.gradient_accumulation_steps) * self.args.num_epoch

        print(
        "trainset: {}, batch_size: {}, gradient_accumulation_steps: {}, num_epoch: {}, num_train_optimization_steps: {}".format(
            len(train_data_loader) * self.args.batch_size, self.args.batch_size, self.args.gradient_accumulation_steps,
            self.args.num_epoch, num_train_optimization_steps))

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        print("Number of parameters:", sum(p[1].numel() for p in param_optimizer if p[1].requires_grad))

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        scheduler = None

        if self.args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model, delay_allreduce=True)
        elif self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.resume is True:
            print("resume from: {}".format(self.args.resume_model))
            output_optimizer_file = os.path.join(self.args.resume_model, "optimizer.pt")
            checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")

            optimizer.load_state_dict(checkpoint_optimizer["optimizer"])

        return model, optimizer, scheduler, train_data_loader, farasa_data_loader, dev_dataloader

    def run(self):
        self.save_args()
        model, optimizer, scheduler, train_data_loader, farasa_data_loader, dev_dataloader = self.prepare_model_optimizer()
        self._train(model, optimizer, scheduler, train_data_loader, farasa_data_loader, dev_dataloader)


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ATB', type=str)
    parser.add_argument('--data_dir', default='ATB', type=str)
    parser.add_argument('--embedding', default='embedding', type=str)
    parser.add_argument('--encoder', default='bilstm', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--num_epoch_multi_cri', default=20, type=int)
    parser.add_argument('--patient', default=5, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=1024, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--bert_model', default='./bert-large-uncased', type=str)
    parser.add_argument('--outdir', default='./', type=str)
    parser.add_argument('--tool', default='stanford', type=str)
    parser.add_argument('--warmup_proportion', default=0.06, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--loss_scale', default=0, type=int)
    parser.add_argument('--multi_criteria', action='store_true', help="Using multi criteria")
    parser.add_argument('--adversary', action='store_true', help="Using adversary learning")
    parser.add_argument('--save', action='store_true', help="Whether to save model")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size", type=int, default=1, help="local_rank for distributed training on gpus")
    parser.add_argument('--resume', action='store_true', help="whether load previous checkpint and start training")
    parser.add_argument('--resume_model', default='', type=str)
    args = parser.parse_args()

    args.initializer = torch.nn.init.xavier_uniform_

    return args

def main():
    args = get_args()

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except Exception as e:
            print(str(e))
    args.outdir = os.path.join(args.outdir, "{}_{}_bts_{}_ws_{}_lr_{}_warmup_{}_seed_{}_bert_dropout_{}_{}".format(
        args.tool,
        args.dataset,
        args.batch_size,
        args.world_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        args.bert_dropout,
        now_time
    ))
    if args.save:
        args.outdir = "{}_save".format(args.outdir)
    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except Exception as e:
            print(str(e))

    output_args_file = os.path.join(args.outdir, 'training_args.bin')
    torch.save(args, output_args_file)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    log_file = '{}/{}-{}.log'.format(args.log, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
