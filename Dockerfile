# Use official Node.js LTS image
FROM node:18

# Create app directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy entire project (React + Express)
COPY . .

# Build the React app
RUN npm run build

# Expose port
EXPOSE 8080

# Start the server
CMD [ "node", "server.js" ]
