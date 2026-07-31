FROM node:18-bullseye-slim

# Install Python and audio system dependencies (libsndfile is critical for Librosa/Praat)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/app

# We must use break-system-packages for system-wide pip install in python3.11+
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Install Node dependencies
COPY package*.json ./
RUN npm install --production

# Copy backend source code
COPY . .

# Ensure uploads directory exists
RUN mkdir -p uploads

# Expose API port
EXPOSE 5000

# Start server
CMD ["npm", "start"]
