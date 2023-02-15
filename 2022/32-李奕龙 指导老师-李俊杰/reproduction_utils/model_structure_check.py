from lib2to3.pgen2 import token
from operator import mod
import tempfile
from turtle import pd
from sklearn import model_selection
from sklearn.cluster import ward_tree
import paddle
from canine import CanineTokenizer as PDTokenizer
from canine import CanineModel as PDmodel
from canine import *
import time

paddle.set_device("cpu")

def get_info(func):
    def wrapper(*args, **kwargs):
        time1 = time.time()
        result = func(*args, **kwargs)
        print(
            f">>>>>> Run `{func.__name__}` successfully, time: {time.time() - time1:.2f}s")
        return result
    return wrapper


@get_info
def base_model_check(pd_inputs, pd_model):
    pd_model.eval()

    with paddle.no_grad():
        outputs = pd_model(**pd_inputs)[0]

    print("outputs:", outputs[:, :5, 0])
    return pd_model


@get_info
def qa_model_check(inputs, pd_model):
    model = CanineForQuestionAnswering(pd_model)
    start_logits, end_logits = model(**inputs)
    print(start_logits[:, :5])
    print(end_logits[:, :5])


@get_info
def classification_check(inputs, pd_model):
    model = CanineForSequenceClassification(pd_model)
    print(model(**inputs)[:, :5])


@get_info
def token_classify_check(inputs, pd_model):
    model = CanineForTokenClassification(pd_model)
    print(model(**inputs)[:, :5])


@get_info
def multi_choise_check(pd_model,tokenizer):
    from paddlenlp.data import Pad, Dict

    model = CanineForMultipleChoice(pd_model)
    data = [
        {
            "question": "how do you turn on an ipad screen?",
            "answer1": "press the volume button.",
            "answer2": "press the lock button.",
            "label": 1,
        },
        {
            "question": "how do you indent something?",
            "answer1": "leave a space before starting the writing",
            "answer2": "press the spacebar",
            "label": 0,
        },
    ]

    text = []
    text_pair = []
    for d in data:
        text.append(d["question"])
        text_pair.append(d["answer1"])
        text.append(d["question"])
        text_pair.append(d["answer2"])
    inputs = tokenizer(text, text_pair, 
                            padding="longest", 
                            return_tensors="pd",
                            return_attention_mask=True,
                            return_token_type_ids=True,
                            )
    reshaped_logits = model(**inputs)
    print(reshaped_logits[:, :5])


@paddle.no_grad()
def run_check():
    text = [["question","answer"],["seq1","seq2"]]

    model_name = "canine-s"
    PDtokenizer = PDTokenizer.from_pretrained(model_name)
    model = PDmodel()
    pd_inputs = PDtokenizer(text,
                         padding="longest",
                         return_attention_mask=True,
                         return_token_type_ids=True,
                         return_tensors="pd")

    base_model_check(pd_inputs, model)
    qa_model_check(pd_inputs, pd_model=model)
    classification_check(pd_inputs, model)
    token_classify_check(pd_inputs, model)
    multi_choise_check(model, PDtokenizer)

if __name__ == "__main__":
    run_check()
