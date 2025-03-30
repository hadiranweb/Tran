import streamlit as st
from openrouter import OpenRouter
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

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_text_chunk(text, model_choice, delay=1):
    """ترجمه هر بخش متن با مدیریت خطا و قابلیت تلاش مجدد"""
    time.sleep(delay)  # مدیریت Rate Limit
    
if model_choice == "DeepSeek":
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    prompt = f"متن زیر را به فارسی روان ترجمه کن:\n{text}"
    response = client.chat.create(
        model="deepseek/deepseek-v3-base:free",  # یا مدل دیگری که می‌خواهید از OpenRouter استفاده کنید
        messages=[
            {"role": "system", "content": "تو یک مترجم حرفه‌ای هستی"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
else:
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.chat.create(
        model="openai/gpt-3.5-turbo-0613",  # استفاده از GPT-3.5 از طریق OpenRouter
        messages=[
            {"role": "system", "content": "مترجم حرفه‌ای فارسی"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"خطا در ترجمه: {str(e)}")
        return f"[خطا در ترجمه این بخش: {str(e)}]"

def parallel_translate(chunks, model_choice, max_workers=4):
    """پردازش موازی پیشرفته با مدیریت خطا و تاخیرهای هوشمند"""
    translated_chunks = [None] * len(chunks)
    delay_per_worker = 2  # تاخیر بین درخواست‌ها برای هر worker
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for i, chunk in enumerate(chunks):
            # تاخیرهای متفاوت برای هر worker
            delay = (i % max_workers) * delay_per_worker
            future = executor.submit(
                translate_text_chunk, 
                chunk, 
                model_choice,
                delay
            )
            future_to_index[future] = i
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                translated_chunks[index] = future.result()
                # نمایش پیشرفت
                progress = (sum(1 for x in translated_chunks if x is not None)) / len(chunks)
                st.session_state['progress'] = progress
            except Exception as e:
                translated_chunks[index] = f"خطا در ترجمه: {str(e)}"
                st.error(f"خطا در ترجمه بخش {index + 1}: {str(e)}")
    
    return translated_chunks

def create_pdf(text, filename="ترجمه.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    # استفاده از فونت‌های استاندارد اگر فونت سفارشی وجود ندارد
    try:
        # اضافه کردن فونت‌ها از مسیر کامل
        font_path = os.path.join(os.path.dirname(__file__), 'fonts')
        pdf.add_font('Vazir', '', os.path.join(font_path, 'Vazirmatn-Regular.ttf'))
        pdf.add_font('VazirB', 'B', os.path.join(font_path, 'Vazirmatn-Bold.ttf'))
        pdf.set_font('Vazir', size=12)
    except Exception as e:
        st.warning("فونت فارسی یافت نشد. از فونت استاندارد استفاده می‌شود.")
        pdf.set_font("helvetica", size=12)
    
    pdf.multi_cell(0, 10, text=text, align="R")
    return pdf.output()
    
    pdf.multi_cell(0, 10, text=text, align="R")
    return pdf.output()
    
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
    
    with st.expander("نمایش محتوای استخراج شده"):
        if file_type == "PDF":
            chunks = process_pdf_by_page(uploaded_file)
            for i, chunk in enumerate(chunks[:3]):  # نمایش 3 صفحه اول
                st.text(f"صفحه {i+1}:\n{chunk[:500]}...")  # نمایش بخشی از متن
        elif file_type == "زیرنویس (SRT)":
            chunks = process_srt_file(uploaded_file)
            st.text(f"تعداد خطوط: {len(chunks)}")
            st.text("\n".join(chunks[:5]))  # نمایش 5 خط اول
        else:
            chunks = [uploaded_file.read().decode("utf-8")]
            st.text(chunks[0][:1000] + "...")  # نمایش بخشی از متن
    
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
