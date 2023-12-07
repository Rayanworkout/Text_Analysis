PYTHON = python3
PIP = pip3


install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download fr_core_news_sm
	$(PYTHON) -m spacy download en_core_web_sm

run:
	streamlit run app.py