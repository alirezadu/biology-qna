FROM python:3.13-slim

# نصب وابستگی‌ها
RUN pip install streamlit langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers pypdf

# نصب Ollama
RUN curl https://ollama.ai/install.sh | sh

# کپی فایل‌های اپ
COPY . /app
WORKDIR /app

# نصب مدل phi3:mini
RUN ollama serve & sleep 5 && ollama pull phi3:mini

# اجرای Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
