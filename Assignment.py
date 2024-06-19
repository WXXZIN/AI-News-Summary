# conda create -n assign2 nltk pandas
# conda activate assign2
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# pip install transformers[torch] datasets torch
# pip install scikit-learn beautifulsoup4 selenium webdriver-manager evaluate absl-py rouge-score

import csv
import json
import os
import sys
from glob import glob

import pandas as pd
import requests
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from evaluate import load
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

HEADERS = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
SECTION_MAP = {"정치": "100", "경제": "101", "사회": "102"}

# ==================================================================

# 데이터 전처리
def preprocess_data(): 
    # "REPORT-news_r"로 시작하는 모든 json 파일 찾기
    news_json_files = glob('./origin_data/news_r/REPORT-news_r*.json')

    # "data/literature/" 위치에 있는 모든 json 파일 찾기
    literature_json_files = glob('./origin_data/literature/REPORT-literature*.json')

    data = []

    # 각 json 파일을 읽고, 필요한 정보를 추출
    for json_file in news_json_files + literature_json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            summaries = [json_data['Annotation'].get('summary'+str(i)) for i in range(1, 4) if json_data['Annotation'].get('summary'+str(i)) is not None]
            summary = ' '.join(summaries)

            data.append({
                "type": 'news' if json_data['Meta(Acqusition)'].get('doc_type') == 'news_r' else json_data['Meta(Acqusition)'].get('doc_type'),
                "id": json_data['Meta(Acqusition)'].get('doc_id'),
                "passage": json_data['Meta(Refine)'].get('passage'),
                "summary": summary
            })

    if not os.path.exists('./data'):
        os.makedirs('./data')

    # CSV 파일로 저장
    with open('./data/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["type", "id", "passage", "summary"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

    data = pd.read_csv('./data/data.csv')

    # 데이터셋 분리
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

    # 데이터 저장
    train_data.to_csv('./data/train_data.csv', index=False)
    valid_data.to_csv('./data/valid_data.csv', index=False)

# 데이터 준비
def prepare_data():
    data_files = {'train': './data/train_data.csv', 'valid': './data/valid_data.csv'}
    dataset = load_dataset('csv', data_files=data_files)
    
    return dataset

# 학습
def train_summarize(totalEpochs=3):
    dataset = prepare_data()

    model_name = "gogamza/kobart-base-v2"
    tokenizer = tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2').to(device)

    # 토근나이징
    def tokenize_function(examples):
        inputs = tokenizer(examples['passage'], truncation=True, padding='max_length', max_length=1024)
        targets = tokenizer(examples['summary'], truncation=True, padding='max_length', max_length=128)

        # Important: 'labels' should be the target tokens
        return {
            "input_ids": inputs["input_ids"], 
            "attention_mask": inputs["attention_mask"], 
            "labels": targets["input_ids"]
        }
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=totalEpochs,    # number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=16,   # accumulate gradients over 4 steps
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch"      # evaluation is done at the end of each epoch
    )

    # trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['valid']
    )

    # 모델 학습 시키기
    trainer.train()

    # 학습결과 저장
    model_path = './summary_model/'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# BeautifulSoup 객체 가져오기
def get_soup_obj(url):
    response = requests.get(url, headers=HEADERS)

    return BeautifulSoup(response.text, 'html.parser')

# 드라이버 설정
def get_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')
    service = ChromeService(ChromeDriverManager().install())

    return webdriver.Chrome(service=service, options=chrome_options)

# 네이버 뉴스 정보 가져오기
def get_news_info(section_id):
    soup = get_soup_obj(f"https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1={section_id}")
    
    news_list = []
    list = soup.find('ul', class_='type06_headline').find_all('li')

    for li in list:
        news_info = {
            "title" : li.img.attrs.get('alt') if li.img else li.a.text.replace("\n", "").replace("\t","").replace("\r",""),
            "news_url" : li.a.attrs.get('href'),
        }
        news_list.append(news_info)
    
    return news_list

# 네이버 뉴스 내용 가져오기
def get_news_contents(news_url):
    driver = get_driver()
    driver.get(news_url)
    
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, 'media_end_head_autosummary_button'))).click()
        time.sleep(2)

    except Exception as e:
        print(f"요약봇 버튼을 찾거나 클릭하는 중 오류가 발생했습니다: {e}")
        driver.quit()
        
        return False, None, None
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    body = soup.select_one('#dic_area')
    news_contents = " ".join(content for content in body.stripped_strings) if body else ""
    summary_contents = soup.select_one('div._contents_body._SUMMARY_CONTENT_BODY')

    driver.quit()

    return True, news_contents.strip(), summary_contents.get_text(strip=True) if summary_contents else ""


# 네이버 뉴스 가져오기
def get_naver_news(sec_input=None):
    if sec_input not in SECTION_MAP:
        print(f"존재하지 않는 섹션입니다: {sec_input}")

        return {}
    
    news_list = get_news_info(SECTION_MAP[sec_input])

    if not news_list:
        print(f"뉴스 정보를 가져오는 중 오류가 발생했습니다: {sec_input}")

        return {}

    for news_info in news_list:
        print(news_info['title'])
        print(news_info['news_url'])

        try:
            button_found, news_contents, summary_contents = get_news_contents(news_info['news_url'])

            if not button_found:
                print(f"요약봇 버튼을 찾거나 클릭하는 중 오류가 발생했습니다: {news_info['news_url']}")
                
                continue

            news_info.update({'news_contents': news_contents, 'summary_contents': summary_contents})
   
            return {sec_input: news_info}
   
        except Exception as e:
            print(f"뉴스 내용을 가져오는 중 오류가 발생했습니다: {e}")
   
            continue
    
    print(f"뉴스 내용을 가져오는 중 오류가 발생했습니다: {sec_input}")
   
    return {}

# 요약
def summarize(sec_input: str, model_path: str) -> str:
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    news_info = get_naver_news(sec_input)
    original_text = news_info[sec_input]['news_contents']
    naver_summary = news_info[sec_input]['summary_contents']
    
    if original_text is not None and isinstance(original_text, str):
        inputs = tokenizer(original_text, truncation=True, padding='max_length', max_length=1024, return_tensors="pt").to(device)
    else:
        print("뉴스 내용이 없습니다.")
    
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=128, early_stopping=True)
    model_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    model_rouge_score = summarize_score(original_text, model_summary)
    naver_rouge_score = summarize_score(original_text, naver_summary)
    model_navaer_similarity_score = similarity_score(model_summary, naver_summary)

    return original_text, model_summary, model_rouge_score, naver_summary, naver_rouge_score, model_navaer_similarity_score

# ROUGE 점수 계산
def summarize_score(reference_text: str, generated_summary: str):
    rouge = load("rouge", trust_remote=True)

    rouge_output = rouge.compute(predictions=[generated_summary], references=[reference_text])

    rouge_scores = {key: value for key, value in rouge_output.items()}
    
    return rouge_scores

# Cosine 유사도 계산
def similarity_score(reference_text: str, generated_summary: str):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([reference_text, generated_summary])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]
    
    return cosine_sim

# ===============================================================================
# main, train or summary
# ===============================================================================
if __name__ == "__main__":
    print('\n', '='*80)
    print('Usage [Training]: python Assignment.py   train   totalEpochNo[예:10]')
    print('Usage [Summary]: python Assignment.py   summary   ')

    if len(glob('./data/*.csv')) != 3:
        print("Data preprocessing...")
        preprocess_data()

    if len(sys.argv) > 1:
        # ---------------------------------- 학습
        if sys.argv[1] == 'train':
            totalEpochs = int(sys.argv[2])
            train_summarize(totalEpochs)

        # ---------------------------------- 요약
        elif sys.argv[1] == 'summary':
            while True:
                sec_input = input("\n*** Type a section (e.g., '정치', '경제', '사회') or 'quit' to stop: ").strip()
                if sec_input == 'quit':
                    break

                original_text, model_summary, model_rouge_score, naver_summary, naver_rouge_score, model_navaer_similarity_score = summarize(sec_input, './summary_model/')

                print("\n*** Original Text: ", original_text)
                print("\n*** Model Summary: ", model_summary)
                print("\n*** Model ROUGE Score: ", model_rouge_score)
                print("\n*** Naver Summary: ", naver_summary)
                print("\n*** Naver ROUGE Score: ", naver_rouge_score)
                print("\n*** Model-Naver Similarity Score: ", model_navaer_similarity_score)

    else:
        print("Usage: python Assignment.py train|summary")
        exit()