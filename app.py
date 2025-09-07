import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_grok import Grok
from langchain.chains import RetrievalQA
import os

# تنظیمات
BOOKS_DIR = "books"
BOOKS = ["zist10.pdf", "zist11.pdf", "zist12.pdf"]

# لود و index کتاب‌ها
@st.cache_resource
def load_books():
    docs = []
    for book in BOOKS:
        book_path = os.path.join(BOOKS_DIR, book)
        if os.path.exists(book_path):
            loader = PyPDFLoader(book_path)
            docs.extend(loader.load())
        else:
            st.error(f"فایل {book} پیدا نشد! مطمئن شو تو فولدر books آپلود شده.")
    if not docs:
        st.error("هیچ PDFای لود نشد!")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# لود vectorstore
try:
    vectorstore = load_books()
    if vectorstore is None:
        st.error("کتاب‌ها لود نشدند. لطفاً فایل‌های PDF رو چک کن.")
        st.stop()
except Exception as e:
    st.error(f"خطا در لود کتاب‌ها: {e}")
    st.stop()

# لود API Grok
try:
    api_key = os.getenv("XAI_API_KEY")  # API Key از متغیر محیطی
    if not api_key:
        st.error("لطفاً API Key را در Streamlit Cloud تنظیم کنید!")
        st.stop()
    llm = Grok(api_key=api_key, model="grok-3")
except Exception as e:
    st.error(f"خطا در لود Grok API: {e}")
    st.stop()

# زنجیره QA با RAG
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False
    )
except Exception as e:
    st.error(f"خطا در ساخت زنجیره QA: {e}")
    st.stop()

# رابط کاربری Streamlit
st.title("وب اپ Q&A زیست‌شناسی دبیرستان")
st.subheader("سوال زیستی یا تست بنویس، جواب دقیق با توضیح کامل می‌دم!")
st.write("بر اساس کتاب‌های زیست دهم، یازدهم، دوازدهم")

# ورودی سوال
question = st.text_area("سوال یا تست (عادی/شمارشی) رو بنویس:", height=150)
if st.button("پیدا کردن جواب"):
    if question:
        with st.spinner("در حال تحلیل..."):
            try:
                response = qa_chain.run(f"""تو متخصص زیست‌شناسی دبیرستان ایران هستی. بر اساس کتاب‌های زیست دهم، یازدهم، دوازدهم، به این سوال جواب بده. اگر تست است، گزینه درست رو مشخص کن و توضیح کامل بده که چرا درست است و گزینه‌های غلط چرا غلطن. جواب باید دقیق، کامل و بدون اشکال باشه. سوال: {question}""")
                st.markdown(response)
            except Exception as e:
                st.error(f"خطا: {e}. لطفاً دوباره امتحان کن یا سوال رو تغییر بده.")
    else:
        st.warning("لطفاً یه سوال یا تست بنویس!")
