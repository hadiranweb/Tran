import streamlit as st
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient
import openai
import time
import os
from dotenv import load_dotenv
from fpdf import FPDF
import tempfile
import concurrent.futures
from functools import partial
import json

# --- تنظیمات اولیه ---
load_dotenv()
st.set_page_config(page_title="مترجم هوشمند حرفه‌ای", layout="wide")
st.title("📖 مترجم پیشرفته PDF و زیرنویس")

# راست‌چین کردن متن فارسی
st.markdown("""
<style>
@font-face {
    font-family: 'Vazir';
    src: url('fonts/Vazirmatn-Regular.ttf') format('truetype');
}

body, p, div, h1, h2, h3, h4, h5, h6 {
    font-family: 'Vazir', 'B Nazanin', Tahoma, sans-serif !important;
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# --- توابع اصلی ---
def process_pdf_by_page(uploaded_file):
    """استخراج متن صفحه‌به‌صفحه از PDF"""
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return [page.get_text() for page in doc]

def process_srt_file(uploaded_file):
    """پردازش فایل زیرنویس"""
    content = uploaded_file.read().decode("utf-8")
    blocks = content.split('\n\n')
    return [block.split('\n')[2] for block in blocks if len(block.split('\n')) >= 3]

def translate_text_chunk(text, model_choice, delay=1):
    """ترجمه هر بخش متن با تاخیر"""
    time.sleep(delay)  # مدیریت Rate Limit
    
    if model_choice == "DeepSeek":
        client = InferenceClient(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            token=os.getenv("HUGGINGFACE_TOKEN"))
        prompt = f"متن زیر را به فارسی روان ترجمه کن:\n{text}"
        return client.text_generation(prompt, max_new_tokens=2000)
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "مترجم حرفه‌ای فارسی"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content

def parallel_translate(chunks, model_choice, max_workers=4):
    """پردازش موازی با ThreadPool"""
    translated_chunks = [None] * len(chunks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(translate_text_chunk, chunk, model_choice): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                translated_chunks[index] = future.result()
            except Exception as e:
                translated_chunks[index] = f"خطا در ترجمه: {str(e)}"
    
    return translated_chunks

def create_pdf(text, filename="ترجمه.pdf"):
    """ایجاد PDF از متن ترجمه‌شده"""
    pdf = FPDF()
    pdf.add_page()
    
  try:
    pdf.add_font('Vazir', '', 'fonts/Vazirmatn-Regular.ttf')  # حذف پارامتر uni
    pdf.add_font('VazirB', 'B', 'fonts/Vazirmatn-Bold.ttf')  # حذف پارامتر uni
    pdf.set_font('Vazir', size=12)
except Exception as e:
    st.warning(f"خطا در بارگذاری فونت: {str(e)}")
    pdf.add_font("Arial", "", "arial.ttf")  # مشخص کردن مسیر فایل فونت
    pdf.set_font("Arial", size=12)
    
    pdf.multi_cell(0, 10, txt=text, align="R")
    return pdf.output(dest='S').encode('latin1')
    
# --- رابط کاربری ---
with st.sidebar:
    st.header("تنظیمات پیشرفته")
    file_type = st.radio(
        "نوع فایل ورودی:",
        ["PDF", "زیرنویس (SRT)", "متن ساده"]
    )
    
    model_choice = st.radio(
        "مدل ترجمه:",
        ["DeepSeek", "OpenAI"]
    )
    
    parallel_enabled = st.checkbox("فعال‌سازی پردازش موازی", value=True)
    max_workers = st.slider("تعداد Threadها", 1, 8, 4) if parallel_enabled else 1

uploaded_file = st.file_uploader(
    "فایل خود را آپلود کنید", 
    type=["pdf", "txt", "srt"],
    help="PDF, فایل متنی یا زیرنویس SRT"
)

if uploaded_file:
    st.success(f"✅ فایل {uploaded_file.name} با موفقیت آپلود شد!")
    
    # استخراج محتوا بر اساس نوع فایل
    if file_type == "PDF":
        chunks = process_pdf_by_page(uploaded_file)
    elif file_type == "زیرنویس (SRT)":
        chunks = process_srt_file(uploaded_file)
    else:
        chunks = [uploaded_file.read().decode("utf-8")]
    
    st.info(f"🔍 {len(chunks)} بخش قابل پردازش شناسایی شد")

    if st.button("شروع ترجمه", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_area = st.empty()
        
        start_time = time.time()
        
        if parallel_enabled and len(chunks) > 1:
            translated = parallel_translate(chunks, model_choice, max_workers)
        else:
            translated = []
            for i, chunk in enumerate(chunks):
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)
                status_text.text(f"در حال ترجمه بخش {i + 1}/{len(chunks)}...")
                translated.append(translate_text_chunk(chunk, model_choice))
        
        # نمایش نتایج
        full_translation = "\n\n".join(translated)
        result_area.text_area("ترجمه کامل", full_translation, height=400)
        
        # اطلاعات پردازش
        elapsed_time = time.time() - start_time
        st.success(f"⏱️ ترجمه کامل شد! زمان پردازش: {elapsed_time:.2f} ثانیه")
        
        # دانلود نتایج
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "دانلود ترجمه (PDF)",
                data=create_pdf(full_translation),
                file_name=f"ترجمه_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf"
            )
        with col2:
            st.download_button(
                "دانلود ترجمه (TXT)",
                data=full_translation,
                file_name=f"ترجمه_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )

        # ذخیره در حافظه موقت
        st.session_state['last_translation'] = full_translation
