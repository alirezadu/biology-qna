FROM python:3.13-slim

# نصب وابستگی‌ها
RUN pip install streamlit langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers transformers pypdf

# کپی فایل‌های اپ
COPY . /app
WORKDIR /app

# اجرای Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
