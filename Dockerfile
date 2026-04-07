FROM python:3.11-slim

# Hugging Face Spaces requires port 7860
ENV PORT=7860
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose port
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "mood_regulator.main:app", "--host", "0.0.0.0", "--port", "7860"]