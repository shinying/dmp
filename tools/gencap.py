from argparse import ArgumentParser
import os
import re

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, BloomForCausalLM
import torch


def prompt(word):
    return f"Provide a caption for a photo of a {word}. The caption should contain many adjectives, should describe colors, styles, lighting and materials in the photo, should be in English and should be no longer than 150 characters. Caption:"


def clean_l_sentences(ls):
    s = [re.sub('[\d\n]', '', x) for x in ls]
    # s = [x.replace(".","").replace("-","").replace(")","").strip() for x in s]
    return s


def flant5xl_compute_word2sentences(word, num=100):
    text_input = prompt(word)
    l_sentences = []
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids.to("cuda")
    input_length = input_ids.shape[1]
    while True:
        try:
            outputs = model.generate(input_ids,temperature=0.95, num_return_sequences=16, do_sample=True, max_length=128, min_length=15, eta_cutoff=1e-5)
            # output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except:
            continue
        for line in output:
            line = line.strip()
            line = "A DSLR photo, " + line[0].lower() + line[1:]
            # print(line)
            l_sentences.append(line)
        print(len(l_sentences))
        if len(l_sentences)>=num:
            break
    l_sentences = clean_l_sentences(l_sentences)

    return l_sentences


def bloomz_compute_sentences(words, num=100):
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
    model = BloomForCausalLM.from_pretrained("bigscience/bloomz-7b1", device_map="auto", torch_dtype=torch.float16)

    all_result = []
    for word in words:
        word_result = []
        text_input = prompt(word)
        input_ids = tokenizer(text_input, return_tensors="pt").input_ids.to("cuda")
        input_length = input_ids.shape[1]
        t = 0.95
        eta = 1e-5
        min_length = 15

        while True:
            try:
                outputs = model.generate(input_ids,temperature=t, num_return_sequences=16, do_sample=True, max_length=128, min_length=min_length, eta_cutoff=eta)
                output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
            except:
                continue
            output = [line.strip() for line in output]
            word_result.extend(output)
            print(output)

            if len(word_result) >= num:
                word_result = word_result[:num]
                break

        all_result += word_result
    all_result = clean_l_sentences(all_result)

    return all_result


parser = ArgumentParser()
parser.add_argument('word')
parser.add_argument('-n', '--num', default=10000, type=int)
parser.add_argument('-o', '--output')
args = parser.parse_args()

if os.path.isfile(args.word):
    args.word = open(args.word).read().splitlines()
else:
    args.word = [args.word]

# sens = flant5xl_compute_word2sentences(args.word, num=args.num)
sens = bloomz_compute_sentences(args.word, num=args.num)
if not args.output:
    os.makedirs('captions', exist_ok=True)
    name = args.word[0] if len(args.word) == 1 else "all"
    args.output = f'captions/{name}.txt'
with open(args.output, 'w') as f:
    for sen in sens:
        print(sen, file=f)
