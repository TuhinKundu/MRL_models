import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template


def zero_shot_classifier(model, classnames, templates, dim, args, device):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(device)  # tokenize
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            if args.force_mrl_loss:
                class_embeddings = class_embeddings[:,:dim] # consider only dim for creating zeroshot classifier 
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, dim, args, device, cast_dtype):
    autocast = get_autocast(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(device)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                if args.force_mrl_loss:
                    image_features = image_features[:,:dim]
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, device, cast_dtype):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    # if MRL 
    if args.force_mrl_loss:
        results = {}
        for dim in args.mrl_dim_to_consider: 
            logging.info(f'Building zero-shot classifier dim-{dim}')
            classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, dim, args, device)

            logging.info('Using classifier')
            if 'imagenet-val' in data:
                top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, dim, args, device, cast_dtype)
                top1_name = f'imagenet-zeroshot-val-d{dim}-top1'
                top5_name = f'imagenet-zeroshot-val-d{dim}-top5'
                results[top1_name] = top1
                results[top5_name] = top5
            if 'imagenet-v2' in data:
                top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, dim, args, device, cast_dtype)
                top1_name = f'imagenet-zeroshot-val-d{dim}-top1'
                top5_name = f'imagenet-zeroshot-val-d{dim}-top5'
                results[top1_name] = top1
                results[top5_name] = top5
    
    # for other losses than MRL
    else: 
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, 0, args, device)

        logging.info('Using classifier')
        results = {}
        if 'imagenet-val' in data:
            top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, 0, args, device, cast_dtype)
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5
        if 'imagenet-v2' in data:
            top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, 0, args, device, cast_dtype)
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5

    # if not MRL 
    logging.info('Finished zero-shot imagenet.')
    return results
