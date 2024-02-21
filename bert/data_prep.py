from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import multiprocessing
from itertools import chain

def tokenize_input(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = DATA_COLLATOR(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}



# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
raw_datasets = concatenate_datasets([bookcorpus, wiki])
#raw_datasets = bookcorpus
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
num_proc = multiprocessing.cpu_count()
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

tokenized_datasets = raw_datasets.map(tokenize_input, batched=True, remove_columns=["text"], num_proc=num_proc)
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)


#Masking collator
DATA_COLLATOR = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)
processed_dataset = tokenized_datasets.map(insert_random_mask, batched=True, num_proc=num_proc,
                                           remove_columns=tokenized_datasets.column_names)






processed_dataset.save_to_disk('wiki_bookcorpus_train')

