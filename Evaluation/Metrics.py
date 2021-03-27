from tqdm import tqdm
import torch
from datasets import load_metric
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu


def perplexity(model, dataloader):
    eval_loss = 0
    with tqdm(total=len(dataloader)) as pbar:
        for idx, entry in enumerate(dataloader):
            with torch.no_grad():
                inputs = entry[0].cuda()
                attn_masks = entry[1].cuda()
                labels = entry[2].cuda()
                outputs = model(inputs, labels=labels, attention_mask=attn_masks)
            loss = outputs[0]
            eval_loss += loss.mean().item()
            pbar.update(2)
    final_eval_loss = eval_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(final_eval_loss))
    return perplexity


def bleurt(target_texts, output_texts):
    metric = load_metric("bleurt")
    tf.compat.v1.flags.DEFINE_string('f', '', '')

    scores = metric.compute(predictions=output_texts, references=target_texts)['scores']
    return scores


def bleu(target_texts, output_texts):
    scores = []
    for idx in range(len(output_texts)):
        reference = [target_texts[idx].split()]
        candidate = output_texts[idx].split()
        scores.append(sentence_bleu(reference, candidate))
    return scores
