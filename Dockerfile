FROM python:3.11-slim
WORKDIR /app
RUN pip install scikit-learn==1.4.1.post1 numpy==1.25.2 pandas==2.2.1 joblib==1.3.2 Flask==3.0.2
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
