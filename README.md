# مترجم هوشمند

پروژه ترجمه خودکار اسناد با قابلیت‌های:
- پردازش موازی صفحات
- پشتیبانی از PDF و زیرنویس
- خروجی PDF با فرمت‌بندی فارسی

## راه‌اندازی
```bash
pip install -r requirements.txt
streamlit run app.py

```
### 5. فایل `Dockerfile` (برای داکریزه کردن - اختیاری)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install -r requirements.txt && \
    mkdir -p /app/fonts

CMD ["streamlit", "run", "app.py"]
