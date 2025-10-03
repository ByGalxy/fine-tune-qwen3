from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载已经微调好的模型

final_save_path = "./final_saved_path"

model = AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer = AutoTokenizer.from_pretrained(final_save_path)

# 构建推理pipeline
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "写一个关于春天的诗歌"
generated_text = pipe(prompt, max_length=100, num_return_sequences=1)

print("生成的文本:")
print(generated_text[0]['generated_text'])
