FROM python:3.13-slim

# نصب وابستگی‌ها
RUN apt-get update && apt-get install -y curl
RUN pip install streamlit langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers pypdf

# نصب دستی Ollama
RUN curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 -o /usr/local/bin/ollama
RUN chmod +x /usr/local/bin/ollama

# کپی فایل‌های اپ
COPY . /app
WORKDIR /app

# اجرای Streamlit و سرور Ollama
CMD ["/bin/sh", "-c", "/usr/local/bin/ollama serve & sleep 10 && /usr/local/bin/ollama pull phi3:mini && streamlit run app.py --server.port=8501"]
