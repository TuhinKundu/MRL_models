'''
adapted from https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_english.py

'''
import os.path

from mteb import MTEB, MTEB_MAIN_EN
import torch
from transformers import AutoModelForMaskedLM, AutoConfig, AutoModel, AutoTokenizer
import sys
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST_prev = (
    TASK_LIST_RERANKING+

[
    "NFCorpus",
    "NQ",
    "QuoraRetrieval"]
)
TASK_LIST = (

[
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
  ]+
["AmazonCounterfactualClassification","TweetSentimentExtractionClassification"
  ]+TASK_LIST_STS

)



class MyModel():
    def __init__(self, model, tokenizer, dim=768):
        self.model = model
        self.tokenizer = tokenizer
        self.dim = dim


    def encode(self, sentences, **kwargs):
             # from_config(config)

            # msg = model.load_state_dict(ckpt_module, strict=False)
            # print(msg)

            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, max pooling.
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            return sentence_embeddings


    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0][:,:,:self.dim]

        #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model_name = sys.argv[1]
config = AutoConfig.from_pretrained(model_name)
tensors = {}



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
if os.path.isfile(sys.argv[2]):
    ckpt = torch.load(sys.argv[2])
    ckpt_module = {}

    for key in ckpt['module'].keys():
        if 'bert.' in key:
            new_key = key.replace('bert.', '')
            ckpt_module[new_key] = ckpt['module'][key]
    msg = model.load_state_dict(ckpt_module, strict=False)

mrl = sys.argv[3]
if mrl.lower() == 'true':
    mrl = True
else:
    mrl = False
print(mrl)
for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = MTEB(
        tasks=[task], task_langs=["en"],
    )

    try:
        if mrl:
            for dim in [768, 384, 192, 96, 48]:
                custom_model = MyModel(model, tokenizer, dim=dim)
                results = evaluation.run(custom_model, eval_splits=eval_splits, output_folder=f"mteb_results/{dim}/{sys.argv[2].split('/')[-1].split('.')[0]}")
        else:
            custom_model = MyModel(model, tokenizer)

             # Remove "en" for running all languages
            results = evaluation.run(custom_model, batch_size=256,
                                     eval_splits=eval_splits, output_folder=f"mteb_results/768/{sys.argv[2].split('/')[-1].split('.')[0]}")
    except:
        print(task, ' not processed')
