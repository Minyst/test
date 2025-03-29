!pip install torch==2.4.0 transformers==4.45.1 datasets==3.0.1 accelerate==0.34.2 trl==0.11.1 peft==0.13.0 deepspeed==0.15.4

import os
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime
from webdriver_manager.chrome import ChromeDriverManager
import json

service = Service("workspace/test/chromedriver.exe")
driver = webdriver.Chrome(service=service)  # 수정된 부분
driver.implicitly_wait(2)  # 웹페이지 로딩 대기 시간 설정

search_page = 4
sort = 1  # 최신순
keywords = ["프리미어리그", "프리메라리가", "분데스리가", "세리에A", "리그앙", "챔피언스리그", "fa컵"]
data_list = []

for keyword in keywords:
    for i in range(search_page):
        start_num = 1 if i == 0 else (i * 10) + 1
        url = f'https://search.naver.com/search.naver?where=news&query={keyword}&sort={sort}&start={start_num}'

        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        articles = soup.select('ul.list_news > li')
        for article in articles:
            media_name = article.select_one('.info.press').text if article.select_one('.info.press') else "Unknown"
            title = article.select_one('.news_tit').text if article.select_one('.news_tit') else "Unknown"
            content = article.select_one('.dsc_wrap').text if article.select_one('.dsc_wrap') else "Unknown"
            data_list.append({"media_name": media_name, "title": title, "content": content})

with open("output.json", "w", encoding="utf-8") as file:
    crawled_json = json.dump(data_list, file, indent=4, ensure_ascii=False)

driver.quit()

# 만약 max_seq_length가 전역 변수라면 정의
max_seq_length = 1024  # 환경에 맞게 수정

# 1. 허깅페이스 허브에서 데이터셋 로드
dataset = load_dataset("json", data_files="output.json")

# 2. system_message 정의
system_message = """

주어진 뉴스 기사 데이터를 변환해야 한다. 
'media_name'과 'title'을 입력으로 사용하고 'content'를 출력으로 사용하라. 
사용자는 기사 제목을 제공하며, 너는 해당 기사 내용을 요약하여 응답해야 한다.
다음의 지시사항을 따르십시오.

검색 결과:
-----
{search_result}"""

# 3. 원본 데이터의 type별 분포 출력 
print("원본 데이터의 type 분포:")
for type_name in set(dataset['type']):
    print(f"{type_name}: {dataset['type'].count(type_name)}")

# 4. train/test 분할 비율 설정 (0.5면 5:5로 분할)
test_ratio = 0.4

train_data = []
test_data = []

# 5. type별로 순회하면서 train/test 데이터 분할
for type_name in set(dataset['type']):
    curr_type_data = [i for i in range(len(dataset)) if dataset[i]['type'] == type_name]
    test_size = int(len(curr_type_data) * test_ratio)
    test_data.extend(curr_type_data[:test_size])
    train_data.extend(curr_type_data[test_size:])

# 6. OpenAI format으로 데이터 변환을 위한 함수 
def format_data(sample):
    search_result = "\n-----\n".join([f"문서{idx + 1}: {result}" for idx, result in enumerate(sample["search_result"])])
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(search_result=search_result),
            },
            {
                "role": "user",
                "content": f"기사 제목: {sample['title']} (출처: {sample['media_name']})",
            },
            {
                "role": "assistant",
                "content": sample["content"]
            },
        ],
    }

# 7. 분할된 데이터를 OpenAI format으로 변환
train_dataset_list = [format_data(dataset[i]) for i in train_data]
test_dataset_list = [format_data(dataset[i]) for i in test_data]

print(f"\n전체 데이터 분할 결과: Train {len(train_dataset_list)}개, Test {len(test_dataset_list)}개")

print("\n학습 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in train_data if dataset[i]['type'] == type_name)
    print(f"{type_name}: {count}")

print("\n테스트 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in test_data if dataset[i]['type'] == type_name)
    print(f"{type_name}: {count}")

# 로컬에 저장된 데이터셋이 있으면 불러오고, 없으면 새로 변환 후 저장
if os.path.exists("train_dataset"):
    train_dataset = load_from_disk("train_dataset")
    print("로컬 train_dataset 로드 완료.")
else:
    # 리스트 형태에서 다시 Dataset 객체로 변경
    train_dataset = Dataset.from_list(train_dataset_list)
    train_dataset.save_to_disk("train_dataset")
    print("train_dataset을 로컬에 저장함.")

if os.path.exists("test_dataset"):
    test_dataset = load_from_disk("test_dataset")
    print("로컬 test_dataset 로드 완료.")
else:
    # 리스트 형태에서 다시 Dataset 객체로 변경
    test_dataset = Dataset.from_list(test_dataset_list)
    test_dataset.save_to_disk("test_dataset")
    print("test_dataset을 로컬에 저장함.")

# 허깅페이스 모델 ID
model_id = "meta-llama/Llama-3.1-8B" 

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto", DeepSpeed로 멀티 GPU 학습을 할 때는 Accelerate나 DeepSpeed가 직접 모델의 분산 배치를 관리하도록 해야하므로 해당 코드를 주석 처리할 것.
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 템플릿 적용 예시
text = tokenizer.apply_chat_template(
    train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False
)
print("템플릿 적용 결과:", text)

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="meta-llama/Llama-3.1-8B",           # 저장될 디렉토리 및 저장소 ID
    num_train_epochs=3,                      # 에포크 수
    per_device_train_batch_size=2,           # GPU 당 배치 크기
    gradient_accumulation_steps=4,           # 그래디언트 누적 스텝
    gradient_checkpointing=True,             # 메모리 절약 체크포인팅
    optim="adamw_torch_fused",               # 최적화기
    logging_steps=10,                        # 로그 주기
    save_strategy="steps",                   # 저장 전략
    save_steps=50,                           # 저장 주기
    bf16=True,                               # bfloat16 사용
    learning_rate=1e-4,                      # 학습률
    max_grad_norm=0.3,                       # 그래디언트 클리핑
    warmup_ratio=0.03,                       # 워밍업 비율
    lr_scheduler_type="constant",            # 학습률 스케줄러 타입
    push_to_hub=False,                       # 허브 업로드 여부
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=None,
    deepspeed="ds_config.json"               # DeepSpeed 설정 파일 지정
)

def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for example in batch:
        # 메시지를 정리 (role과 content만 포함)
        clean_messages = []
        for message in example["messages"]:
            clean_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        # tokenizer.apply_chat_template() 사용
        text = tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,  # 직접 토큰화할 것이므로 False 설정
            add_generation_prompt=False
        ).strip()
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [-100] * len(input_ids)
        
        # LLaMA 모델의 프롬프트 관련 토큰 설정
        inst_start = tokenizer.encode("[INST]", add_special_tokens=False)
        inst_end = tokenizer.encode("[/INST]", add_special_tokens=False)
        assistant_end = tokenizer.encode("</s>", add_special_tokens=False)
        
        # 라벨 생성 (어시스턴트 응답 부분만 학습)
        i = 0
        while i < len(input_ids):
            if (i + len(inst_start) <= len(input_ids) and 
                input_ids[i:i+len(inst_start)] == inst_start):
                inst_pos = i + len(inst_start)
                while inst_pos < len(input_ids):
                    if (inst_pos + len(inst_end) <= len(input_ids) and 
                        input_ids[inst_pos:inst_pos+len(inst_end)] == inst_end):
                        assistant_pos = inst_pos + len(inst_end)
                        while assistant_pos < len(input_ids):
                            if (assistant_pos + len(assistant_end) <= len(input_ids) and 
                                input_ids[assistant_pos:assistant_pos+len(assistant_end)] == assistant_end):
                                for j in range(len(assistant_end)):
                                    labels[assistant_pos + j] = input_ids[assistant_pos + j]
                                break
                            labels[assistant_pos] = input_ids[assistant_pos]
                            assistant_pos += 1
                        break
                    inst_pos += 1
                i = inst_pos
            i += 1
        
        # 배치에 추가
        new_batch["input_ids"].append(input_ids)
        new_batch["attention_mask"].append(attention_mask)
        new_batch["labels"].append(labels)
    
    # 패딩 적용 (최대 길이에 맞추기)
    max_length = max(len(ids) for ids in new_batch["input_ids"])
    for i in range(len(new_batch["input_ids"])):
        padding_length = max_length - len(new_batch["input_ids"][i])
        new_batch["input_ids"][i].extend([tokenizer.pad_token_id] * padding_length)
        new_batch["attention_mask"][i].extend([0] * padding_length)
        new_batch["labels"][i].extend([-100] * padding_length)
    
    # 텐서 변환
    for k, v in new_batch.items():
        new_batch[k] = torch.tensor(v)
    
    return new_batch

def main():
    trainer = SFTTrainer(
        model=model,
        args=args,
        max_seq_length=max_seq_length,  # 최대 시퀀스 길이 설정
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )
    
    # 학습 시작 (모델은 자동으로 허브 및 output_dir에 저장됨)
    trainer.train()
    # 모델 저장 (최종 모델)
    trainer.save_model()

if __name__ == "__main__":
    main()
