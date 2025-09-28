# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Command to run your Flask application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]