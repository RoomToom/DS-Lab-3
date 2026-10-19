# Лабораторна робота №3 — Free Text and NLP

Обробка англомовних текстів: токенізація, стемінг, лематизація, BoW, TF-IDF, Word2Vec та N-грами.

## Завдання

1. `task1_text_cleaning.py` — очищення тексту й поділ на слова.
2. `task2_stemm_lemma.py` — порівняння PorterStemmer і WordNetLemmatizer.
3. `task3_BoW.py` — матриця частот слів.
4. `task4_TFIDF.py` — ваги TF-IDF.
5. `task5_word2vec.py` — навчання Word2Vec і пошук схожих слів.
6. `task6_Ngrams.py` — побудова та підрахунок N-грам.
7. `task7_nlp_pipeline.py` — спільний аналіз документів, ключові слова й косинусна подібність.

## Встановлення

Python 3.11 або 3.12. Команди виконуються з кореня репозиторію.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt', raise_on_error=True); nltk.download('wordnet', raise_on_error=True); nltk.download('omw-1.4', raise_on_error=True)"
```

У Linux/macOS середовище активується командою `source venv/bin/activate`.

## Запуск

```powershell
python -m src.task1_text_cleaning
python -m src.task2_stemm_lemma
python -m src.task3_BoW
python -m src.task4_TFIDF
python -m src.task5_word2vec
python -m src.task6_Ngrams
python -m src.task7_nlp_pipeline
```

Приклади виводу для кожного завдання збережені в `results/task1.txt` — `results/task7.txt`.

Інтерактивний додаток:

```powershell
python -m streamlit run app.py
```

Перевірка:

```powershell
python -m pytest tests/ -v
```

Word2Vec навчається на переданих реченнях. На малому навчальному корпусі знайдені сусіди демонструють роботу методу, але не дають надійної оцінки близькості значень слів.

[Умова лабораторної](https://github.com/UNI-DS/DS-Lab-3) · [Матеріали заняття](https://github.com/mehalyna/UNI-MethDS-Lessons/blob/main/L4_free_text.ipynb)
