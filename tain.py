from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'Qwen3-0.6B-abliterated'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto').to('cuda')

# print('Model and tokenizer loaded successfully.')

from datasets import load_dataset

dataset = load_dataset("json", data_files={"tarn": "dataset/ruozhiba_qa.json"}, split="tarn")
print('数据量：',len(dataset))

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

print(f'train_dataset size: {len(train_dataset)}')
print(f'test_dataset size: {len(test_dataset)}')

print('数据集准备好了')

def tokenize_function(many_samples):
    texts = [f'{instruction}\n{output}' for instruction, output in zip(many_samples['instruction'], many_samples['output'])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding='max_length')
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
print('数据集分词完毕')
# print(len(tokenized_train_dataset), len(tokenized_test_dataset))

# 量化设置
from transformers import BitsAndBytesConfig

quantization_for_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_for_config, trust_remote_code=True, device_map='auto')

print('完成量化模型的加载')

# lora微调设置
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print('完成lora微调设置完毕')

# 训练参数设置, 并进行训练
from transformers import TrainingArguments, Trainer

Training_args = TrainingArguments(
    output_dir='./finetuned_model',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy='steps',
    eval_steps=10,
    learning_rate=3e-5, 
    logging_dir='./logs',
    run_name='Qwen3-0.6B-abliterated-finetuned',
    # dataloader_pin_memory=False,
    # remove_unused_columns=False
)

print('训练参数设置完毕')

# 训练器设置
trainer = Trainer(
    model=model,
    args=Training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

print('开始训练')
trainer.train()
print('训练完成')

# 保存微调后的模型
# 保存lora模型
save_path = './saved_models'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("lora模型已经保存")

# 保存全量模型
final_save_path = './final_saved_path'

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, save_path)
model = model.merge_and_unload()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("全量模型已经保存")
