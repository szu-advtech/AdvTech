import paddle
from transformers import CanineTokenizer
from canine import CanineTokenizer as PDTokenizer

text = "Canine model is tokenization-free."


paddle.set_device("cpu")


def run_check(model_name):
    PDtokenizer = PDTokenizer.from_pretrained(model_name)
    PTtokenizer = CanineTokenizer.from_pretrained('google/canine-s')

    pt_temp = PTtokenizer(text, padding="longest", truncation=True)
    pt_inputs = pt_temp["input_ids"]

    pd_temp = PDtokenizer(text, padding="longest", truncation=True)
    pd_inputs = pd_temp["input_ids"]
    print(">>> running tokenizer check")
    print("pd inputs:", pd_inputs)
    print("pt inputs:", pt_inputs)
    print(f"torch token matched paddle? {pt_inputs == pd_inputs}")

    pttokens = PTtokenizer.convert_ids_to_tokens(ids=pt_inputs)
    pdtokens = PDtokenizer.convert_ids_to_tokens(ids=pd_inputs)
    print(f"torch token matched paddle? {pttokens == pdtokens}")


if __name__ == '__main__':
    run_check(model_name='canine-s')
