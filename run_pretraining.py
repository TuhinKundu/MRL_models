'''
References
1. https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
2. https://huggingface.co/blog/pretraining-bert
3. https://towardsdatascience.com/the-concept-of-transformers-and-training-a-transformers-model-45a09ae7fb50
4. https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling
'''
import warnings

from datasets import load_from_disk
import datasets
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
import open_clip
from clip_model.model import CLIP
from clip_model.data import *
import MRL
from utils import utils
from itertools import cycle
import time
from datetime import timedelta
from accelerate import InitProcessGroupKwargs


class PreTrainer:
    def __init__(self, tokenizer, model, args, image_processor=None):
        self.tokenizer = tokenizer
        self.model = model
        self.args = args

        if self.args.clip:
            self.train_image_processor, self.val_image_processor = image_processor
            if not self.args.mrl:
                self.loss_fn = open_clip.loss.ClipLoss()
            else:
                self.loss_fn = MRL.MatryoshkaClipLoss(nesting_dim=self.args.nesting_dim, loss_weights=self.args.loss_weights)

        # to solve NCCL timeout issue and increase timeout limit: from https://github.com/huggingface/accelerate/issues/314#issuecomment-1785782762
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        self.accelerator = accelerate.Accelerator(split_batches=True, kwargs_handlers=[process_group_kwargs]) # change grad acc and precision (mixed/bf16) in deepspeed config


        self.args = utils.check_batch_params(self.args, self.accelerator.num_processes)


        if self.args.mlm:
            self.train_dataloader, self.val_dataloader = self.load_data_for_mlm(self.args.data_path, self.args.global_batch_size, self.args.num_proc)

        elif args.clip:
            self.train_dataloader, self.val_dataloader = self.clip_webdataset_reader(self.args.data_path, num_proc=self.args.num_proc)

        self.args = utils.check_iter_params(self.args, trainloader=self.train_dataloader)

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.lr_scheduler = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=self.args.warmup_steps,
                                          num_training_steps=self.args.total_steps)

        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader = self.accelerator._prepare_deepspeed(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader
        )

        self.dtype = self.model.get_data_types()[0]
        self.args = utils.check_grad_acc_steps(self.args, self.accelerator)

        self.experiment_name = (f'{self.args.model_name.split("/")[-1]}_mrl{self.args.mrl}_'
                                f'bs{self.args.global_batch_size}_lr{"%.1E"%Decimal(self.args.lr)}_warmup{self.args.warmup_steps}_gradacc{self.accelerator.deepspeed_config["gradient_accumulation_steps"]}')

    def clip_batch_collator(self,batch):

        image_batch, text_batch = [], []
        for i, sample in enumerate(batch):
            image_batch.append(sample[0])
            text_batch.append(sample[1])

        image_tensors = torch.stack(image_batch)
        global_batch, batch, C, H, W = image_tensors.size()
        image_tensors = image_tensors.view(global_batch*batch, C, H, W)
        tokenized_tensors = torch.stack(text_batch)
        global_batch, batch, seq_len = tokenized_tensors.size()
        tokenized_tensors = tokenized_tensors.view(global_batch*batch, seq_len)

        image_tensors = image_tensors.to(self.dtype)
        output_dict = {
            'image':image_tensors, #typecasting to align with accelerate/deepspeed
            'text': tokenized_tensors #tokenized seq expected to be int by torch
        }
        return output_dict

    def clip_webdataset_reader(self, location, num_proc):
        location_shards = os.listdir(location)
        location_shards = [location + shard for shard in location_shards]
        num_files = len(location_shards)
        num_shards = num_files//3 #for each shard there are three files, tar, json, parquet. json is used to compute n_samples
        split_location = (num_shards//40)*3 #taking 2.5% shards for eval
        if split_location == 0:
            split_location = 3
        location_shards.sort()
        val_shards = location_shards[:split_location]
        train_shards = location_shards[split_location:]

        args = self.args
        args.train_data = train_shards
        args.val_data = val_shards
        args.workers = self.args.num_proc
        args.world_size = torch.cuda.device_count()
        args.batch_size = self.args.batch_size
        train_dataset, num_batches = get_wds_dataset(args=self.args, preprocess_img=self.train_image_processor,
                                                       is_train=True, tokenizer=self.tokenizer, return_dataset=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.world_size,#webdataset pipeline already handling batch size
                                      collate_fn=self.clip_batch_collator)
        self.args.train_num_batches = num_batches
        val_dataset, num_batches = get_wds_dataset(args=self.args, preprocess_img=self.val_image_processor,
                                                     is_train=False, tokenizer=self.tokenizer, return_dataset=True)

        val_dataloader = DataLoader(val_dataset, batch_size=args.world_size,
                                    collate_fn=self.clip_batch_collator) #num_workers causes possible bug https://github.com/pytorch/pytorch/issues/4507#issuecomment-1112095166
        self.args.val_num_batches = num_batches
        '''
        train_dataset = wds.WebDataset(train_shards).decode('pil')
        train_dataloader = wds.WebLoader(train_dataset, batch_size=global_batch_size,
                                                       collate_fn=lambda x: self.clip_batch_collator(x, is_train=True),
                                                       num_workers=num_proc//2)

        val_dataset = wds.WebDataset(val_shards).decode('pil')
        val_dataloader = wds.WebLoader(val_dataset, batch_size=global_batch_size,
                                                     collate_fn=lambda x: self.clip_batch_collator(x, is_train=False),
                                                     num_workers=num_proc//4)
        '''
        return train_dataloader, val_dataloader


    def load_data_for_mlm(self, data_path, batch_size, num_proc=16):
        dataset = load_from_disk(data_path).with_format('torch')
        dataset = dataset.rename_columns({'masked_input_ids':'input_ids',
                                           'masked_attention_mask': 'attention_mask',
                                          'masked_labels':'labels'})
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])
        dataset = dataset.train_test_split(train_size=len(dataset)-int(0.1*len(dataset)), test_size=int(0.1*len(dataset)))
        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_proc//2)
        val_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, num_workers=num_proc//4)
        return train_dataloader, val_dataloader

    def print_time_log(self, iter, progress_bar):
        time_elapsed = progress_bar.format_dict['elapsed']
        rate = progress_bar.format_dict["rate"]
        remaining = (progress_bar.total - progress_bar.n) / rate if rate and progress_bar.total else 0
        print(f' Iterations {iter} ; elapsed time: {datetime.timedelta(seconds=time_elapsed)} and remaining: {datetime.timedelta(seconds=remaining)}')

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

            self.args.__dict__ = args
            wandb_config = {'learning_rate': self.args.lr}
            wandb_run = wandb.init(project=self.args.experiment_name, resume=True,
                                   config=wandb_config)
            init_step = args['iters']
        else:
            with open(f'{out_dir}/args.json', 'w') as f:
                json.dump(self.args.__dict__, f)


            wandb_config = {'learning_rate': self.args.lr}
            wandb_run = wandb.init(project=self.experiment_name,
                                   config=wandb_config)
            self.args.experiment_name = self.experiment_name
            init_step = 0


        progress_bar = tqdm(range(init_step, self.args.total_steps))
        self.train_dataloader = iter(cycle(self.train_dataloader))

        for current_iter in range(init_step, self.args.total_steps):

            batch = next(self.train_dataloader)



            with self.accelerator.accumulate(self.model):

                outputs = model(**batch)
                if self.args.clip:
                    loss = self.loss_fn(**outputs)
                elif self.args.mlm:
                    loss = outputs.loss

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.update(1)
            if self.accelerator.is_main_process:

                wandb.log({'loss': loss.item(), 'learning_rate': self.optimizer.param_groups[0]['lr']})
                self.args.iters = current_iter

                if current_iter % (self.args.evaluation_interval//4) ==0:
                    self.print_time_log(current_iter, progress_bar)

            self.model.eval()

            if current_iter % self.args.evaluation_interval==0 and current_iter>self.args.evaluation_interval-1: #current_iter so that loaded ckpt doesn't get saved again immediately
                losses = []
                eval_progress = tqdm(self.val_dataloader)
                for step, batch in enumerate(self.val_dataloader):

                    with torch.no_grad():
                        outputs = self.model(**batch)
                        if self.args.clip:
                            loss = self.loss_fn(**outputs)
                            losses.append(self.accelerator.gather(loss.repeat(batch['image'].size()[0])))
                        elif self.args.mlm:
                            loss = outputs.loss
                            losses.append(self.accelerator.gather(loss.repeat(batch['input_ids'].shape[0])))
                        eval_progress.update(1)


                self.accelerator.wait_for_everyone()
                losses = torch.cat(losses)

                if self.accelerator.is_main_process:
                    if self.args.mlm:
                        losses = losses[:len(self.val_dataloader)]
                        mean_loss = torch.mean(losses)
                        perplexity = math.exp(mean_loss)
                        logging_dict = {'perplexity': perplexity, 'val_loss': mean_loss.item()}
                    elif self.args.clip:
                        mean_loss = torch.mean(losses)
                        logging_dict = {'val_loss': mean_loss.item()}
                    wandb.log(logging_dict)
                    self.print_time_log(current_iter, progress_bar)

                    #unwrap_model = self.accelerator.unwrap_model(self.model)
                    #unwrap_model.save_pretrained(weight_dir, save_function = self.accelerator.save)

                # deepspeed x accelerate bug for saving weights,should be called across all processes
                # https://github.com/huggingface/diffusers/issues/2606#issuecomment-1463193509
                self.accelerator.save_state(out_dir)

                if self.accelerator.is_main_process:
                    self.args.lr = self.optimizer.param_groups[0]['lr']

                    artifact = wandb.Artifact(name=self.experiment_name, type='model')
                    artifact.add_dir(out_dir)
                    wandb_run.log_artifact(artifact)

                    with open(f'{out_dir}/args.json', 'w') as f:
                        json.dump(self.args.__dict__, f)

                    if self.args.kube_pvc:
                        os.system(f'rsync -avP {out_dir} {self.args.kube_pvc}')

                # processes with rank>0 move to next iteration and wait for rank=0 causing NCCL timeout issue
                self.accelerator.wait_for_everyone()


        self.accelerator.save_state(out_dir)
        with open(f'{out_dir}/args.json', 'w') as f:
            json.dump(self.args.__dict__, f)

        if self.args.kube_pvc:
            os.system(f'rsync -avP {out_dir} {self.args.kube_pvc}')

        artifact = wandb.Artifact(name=self.experiment_name, type='model')
        artifact.add_dir(out_dir)
        wandb_run.log_artifact(artifact)




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
    parser.add_argument('--mlm', type=str2bool, default=None, help='to trigger MLM pretraining')
    parser.add_argument('--clip', type=str2bool, default=None, help='to trigger CLIP pretraining')

    parser.add_argument('--mrl', type=str2bool, default=False, help='whether to use MRL')
    parser.add_argument('--mrl_efficient', type=str2bool, default=False, help='whether to use MRL efficient')
    parser.add_argument('--nesting_dim', type=str2list, default=[96,192,384,768], help='MRL nesting dim for text transformer models')
    parser.add_argument('--loss_weights', type=str2list, default=[1.,1.,1.,1.], help='weight for each dim in nesting dim')
    parser.add_argument('--global_batch_size', type=int, default=None, help='total batch size across multiple gpus/nodes')
    parser.add_argument('--batch_size', type=int, default=None, help='micro batch size per gpu')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--total_steps', type=int, default=None, help='total number of iterations. set either iterations or epochs')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs. set either iterations or epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--num_proc', type=int, default=multiprocessing.cpu_count(), help='number of processes')
    parser.add_argument('--evaluation_interval', type=int, default=25000)
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb api login key for remote login')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kube_pvc', type=str, default=None, help='path for k8s pvc for permanent storage')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='gradient accumulation steps')
    #parser.add_argument('--num_gpus', type=int, default=None, help='number of gpus')
    args = parser.parse_args()


    random.seed(args.seed)
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    utils.check_train_selection(args)

    if args.mlm:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name.lower())
        config = AutoConfig.from_pretrained(args.model_name.lower())
        if args.mrl:
            config.nesting_dim = args.nesting_dim
            config.loss_weights = args.loss_weights
        else:
            config.nesting_dim = None
            config.loss_weights = None
        config.mrl_efficient = args.mrl_efficient
        model = BertForMaskedLM(config=config)
        trainer = PreTrainer(tokenizer, model, args)
    elif args.clip:
        tokenizer = open_clip.get_tokenizer(args.model_name)
        config = open_clip.get_model_config(args.model_name)
        model, train_image_processor, val_image_processor = open_clip.create_model_and_transforms(args.model_name)

        model = CLIP(embed_dim=config['embed_dim'], vision_cfg=config['vision_cfg'],text_cfg=config['text_cfg'], cast_dtype=torch.bfloat16)
        trainer = PreTrainer(tokenizer, model, args, image_processor=(train_image_processor, val_image_processor))




    trainer.train()



