#Pull original pharmcat image
FROM pgkb/pharmcat:2.15.1

#Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libpq5 \
    gcc \
    && apt-get clean

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Create worker folder in container
RUN mkdir /data
RUN mkdir -p /tmp/pharmcat
RUN mkdir /result
RUN mkdir -p /scripts/helper_scripts

# Copy the worker script into the container

COPY scripts/ /scripts/

# Set entrypoint to python, pass args from CMD
ENTRYPOINT ["python3", "-u", "/scripts/pharmcat_folder_processor.py"]

# Allow arguments to be passed to the script
CMD []