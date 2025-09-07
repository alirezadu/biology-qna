FROM python:3.13-slim

# نصب وابستگی‌ها
RUN pip install streamlit langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers pypdf

# نصب Ollama
RUN curl https://ollama.ai/install.sh | sh

# کپی فایل‌های اپ
COPY . /app
WORKDIR /app

# راه‌اندازی سرور Ollama و نصب مدل phi3:mini
RUN ollama serve & sleep 10 && ollama pull phi3:mini

# اجرای Streamlit
CMD ["sh", "-c", "ollama serve & sleep 5 && streamlit run app.py --server.port=8501"]
