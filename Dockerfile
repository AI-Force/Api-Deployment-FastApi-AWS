FROM tiangolo/uvicorn-gunicorn:python3.7

# Copy function code
COPY main.py requirements.txt /app/

# Install the function's dependencies using file requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The remaining setup is performed by the parent template.
