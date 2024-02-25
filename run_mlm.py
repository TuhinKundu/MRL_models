'''
References
1. https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
2. https://huggingface.co/blog/pretraining-bert
3. https://towardsdatascience.com/the-concept-of-transformers-and-training-a-transformers-model-45a09ae7fb50
4. https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling
'''
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig
from bert.modeling_bert import BertForMaskedLM
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
import argparse
import multiprocessing, os
import accelerate
import math
import wandb
from tqdm.auto import tqdm
from decimal import Decimal
import datetime, json

class MLMTrainer:
    def __init__(self, tokenizer, model, args):
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.lr_scheduler = get_scheduler("linear",optimizer=self.optimizer,num_warmup_steps=self.args.warmup_steps,
                                          num_training_steps=self.args.total_steps)
        self.train_dataloader, self.val_dataloader = self.load_data(self.args.data_path, self.args.batch_size, self.args.num_proc)
        self.accelerator = accelerate.Accelerator(split_batches=True) #grad acc and precision (mixed/bf16) in deepspeed config
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader =  self.accelerator._prepare_deepspeed(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader
        )
        self.experiment_name = (f'{self.args.model_name}_mrl{self.args.mrl}{self.args.mrl_efficient}_'
                                f'bs{self.args.batch_size}_lr{"%.1E"%Decimal(self.args.lr)}_warmup{self.args.warmup_steps}')



    def load_data(self, data_path, batch_size, num_proc=16):
        dataset = load_from_disk(data_path).with_format('torch')
        dataset = dataset.rename_columns({'masked_input_ids':'input_ids',
                                           'masked_attention_mask': 'attention_mask',
                                          'masked_labels':'labels'})
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])
        dataset = dataset.train_test_split(train_size=len(dataset)-int(0.1*len(dataset)), test_size=int(0.1*len(dataset)))
        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_proc//2)
        val_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, num_workers=num_proc//4)
        return train_dataloader, val_dataloader

    def print_time_log(self, iter, progress_bar):
        time_elapsed = progress_bar.format_dict['elapsed']
        rate = progress_bar.format_dict["rate"]
        remaining = (progress_bar.total - progress_bar.n) / rate if rate and progress_bar.total else 0
        print(f'Iterations {iter} ; elapsed time: {datetime.timedelta(seconds=time_elapsed)} and remaining: {datetime.timedelta(seconds=remaining)}')

    def train(self):

        self.model.train()




        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        print(self.args)
        out_dir = f'{self.args.output_path}/{self.experiment_name}'
        os.makedirs(out_dir, exist_ok=True)

        if self.args.resume_train_from:
            self.accelerator.load_state(self.args.resume_train_from)
            with open(f'{out_dir}/args.json', 'r') as f:
                args = json.load(f)
            init_step = args['iter']
            self.args.__dict__ = args
            wandb_config = {'learning_rate': self.args.lr}
            wandb_run = wandb.init(project=self.experiment_name,
                                   config=wandb_config)
            self.args.wandb_runid = wandb.run.id

        else:
            with open(f'{out_dir}/args.json', 'w') as f:
                json.dump(self.args.__dict__, f)
            init_step = 0

            wandb_config = {'learning_rate': self.args.lr}
            wandb_run = wandb.init(project=self.experiment_name,
                                   config=wandb_config)
            self.args.wandb_runid = wandb.run.id

        progress_bar = tqdm(range(init_step, self.args.total_steps))
        for iter, batch in zip(range(init_step, self.args.total_steps), self.train_dataloader):

            outputs = model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            progress_bar.update(1)
            if self.accelerator.is_main_process:

                wandb.log({'loss': loss.item(), 'learning_rate': self.optimizer.param_groups[0]['lr']})
                self.args.iter = iter

                if iter % (self.args.evaluation_interval//4) ==0:
                    self.print_time_log(iter, progress_bar)

            self.model.eval()

            if iter % self.args.evaluation_interval==0 and iter>self.args.evaluation_interval-1: #iter so that loaded ckpt doesn't get saved again immediately
                losses = []
                eval_progress = tqdm(range(len(self.val_dataloader)))
                for step, batch in enumerate(self.val_dataloader):
                    if step==10:
                        break
                    with torch.no_grad():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        losses.append(self.accelerator.gather(loss.repeat(batch['input_ids'].shape[0])))
                        eval_progress.update(1)


                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    losses = torch.cat(losses)
                    losses = losses[:len(self.val_dataloader)]
                    mean_loss = torch.mean(losses)
                    perplexity = math.exp(mean_loss)
                    wandb.log({'perplexity': perplexity, 'val_loss': mean_loss.item()})
                    self.print_time_log(iter, progress_bar)

                    #unwrap_model = self.accelerator.unwrap_model(self.model)
                    #unwrap_model.save_pretrained(weight_dir, save_function = self.accelerator.save)

                    self.accelerator.save_state(out_dir)

                    self.args.lr = self.optimizer.param_groups[0]['lr']
                    artifact = wandb.Artifact(name=self.experiment_name, type='model')
                    artifact.add_dir(out_dir)
                    wandb_run.log_artifact(artifact)

                    with open(f'{out_dir}/args.json', 'w') as f:
                        json.dump(self.args.__dict__, f)


        self.accelerator.save_state(out_dir)
        artifact = wandb.Artifact(name=self.experiment_name, type='model')
        artifact.add_dir(out_dir)
        wandb_run.log_artifact(artifact)

        with open(f'{out_dir}/args.json', 'w') as f:
            json.dump(self.args.__dict__, f)


if __name__ == '__main__':

    def str2bool(flag):
        if isinstance(flag, bool):
            return flag
        elif flag.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif flag.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean answer expected.')

    def str2list(dims):
        return dims.split[',']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to processed mlm dataset', default='/tk/bert/bookcorpus_train')
    parser.add_argument('--output_path', type=str, help='path to save logs and model weights', default='/tk/output_dir')
    parser.add_argument('--resume_train_from', type=str, help='path to saved checkpoint to resume training', default=None)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--mrl', type=str2bool, default=False, help='whether to use MRL')
    parser.add_argument('--mrl_efficient', type=str2bool, default=False)
    parser.add_argument('--nesting_dim', type=str2list, default=[96,192,384,768])
    parser.add_argument('--batch_size', type=int, default=256, help='total batch size across multiple gpus/nodes')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--total_steps', type=int, default=1000000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_proc', type=int, default=multiprocessing.cpu_count(), help='number of processes')
    parser.add_argument('--evaluation_interval', type=int, default=25000)
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb api login key for remote login')
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name.lower())


    config = AutoConfig.from_pretrained(args.model_name.lower())
    if args.mrl:
        config.nesting_dim = args.nesting_dim
    else:
        config.nesting_dim = None
    config.mrl_efficient = args.mrl_efficient


    model = BertForMaskedLM(config=config)


    trainer = MLMTrainer(tokenizer, model, args)
    trainer.train()



