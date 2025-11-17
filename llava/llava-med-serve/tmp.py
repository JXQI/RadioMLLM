from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "/home/tx-deepocean/data1/jxq/code/MMedAgent_origin/data/llava-med-v1.5-mistral-7b",
    model_base=None,
    model_name="llava-med-v1.5-mistral-7b",
    device="cuda",
)
model = model.to(0)
print(model)
print(model.device)
while True:
    pass
