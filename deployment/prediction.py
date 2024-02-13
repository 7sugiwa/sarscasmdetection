from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./my_fine_tuned_bert')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sarcasm(headline):
    inputs = tokenizer.encode_plus(
        headline,
        None,
        add_special_tokens=True,
        max_length=256,
        pad_to_max_length=True,
        return_token_type_ids=True
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        outputs = model(ids, mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction
