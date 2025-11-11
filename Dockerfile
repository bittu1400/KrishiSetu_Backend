# 1. Use a lightweight Python image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy project files into container
COPY . .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose the port your app will run on
EXPOSE 8080

# 6. Command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

