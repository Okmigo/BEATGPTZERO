# Use Node.js base image
FROM node:18

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy all app source
COPY . .

# Expose Cloud Run port
ENV PORT=3000
EXPOSE 3000

# Start app
CMD ["node", "index.js"]
