# Use an official Python runtime as a base image
FROM python:3.9-slim-buster

RUN useradd -ms /bin/bash python-api
USER python-api

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -y openjdk-8-jdk python3-venv

ENV JAVA_HOME /usr/lib/jvm/java
ENV PATH $JAVA_HOME/bin:$PATH
# Define environment variable
# ENV AWS_ACCESS_KEY_ID=your_access_key_id
# ENV AWS_SECRET_ACCESS_KEY=your_secret_access_key

EXPOSE 8080

# Run app.py when the container launches
ENTRYPOINT ["python", "ETL_project.py"]
CMD ["run", "ETL_project.py"]