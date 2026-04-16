FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files 
# and ensure output is sent straight to the terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (needed for compiling certain python packages like scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY myproject/requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download required nltk datasets during build
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('words')"

# Copy the Django project
COPY myproject /app/

# Expose the default Django port
EXPOSE 8000

# Start Gunicorn or the Django development server
# For development purposes, we use runserver. You might want to switch to gunicorn for production.
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
