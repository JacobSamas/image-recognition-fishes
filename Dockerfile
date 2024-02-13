# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:2.15.0

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
