# AWS EC2 (t3.large) Deployment Guide

This guide contains the exact terminal commands required to deploy the Real Estate ML Tree application to your new AWS EC2 `t3.large` instance. 
The application will run on port `8080` with strict resource limits (1GB RAM, 0.5 CPU) to ensure it behaves as a good neighbor to your OpenStreetMap routing application.

## 1. Update System & Install Dependencies

Connect to your EC2 instance via SSH and run the following commands to ensure your package manager is up to date and git is installed:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y git curl
```

## 2. Install Docker & Docker Compose

Run the official Docker installation script to install the latest version of Docker and its plugins (including Docker Compose):

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```
> [!NOTE]
> You may need to log out and log back in (or run `newgrp docker`) for the group changes to take effect without needing `sudo` for docker commands.

## 3. Clone the Repository

Clone your project from GitHub into the server:

```bash
git clone https://github.com/AybarsOnurlu/RealEstate-ML-Tree.git
cd RealEstate-ML-Tree
```

## 4. Setup Environment Variables

The application requires your Supabase credentials to function. Create the `.env` file using the following command:

```bash
nano .env
```

Paste your Supabase credentials into the editor:
```env
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```
*(Press `Ctrl+O`, `Enter` to save, and `Ctrl+X` to exit nano)*

## 5. Build and Deploy

Finally, use Docker Compose to build the architecture. The `docker-compose.yml` has been specifically tuned for AMD64 with resource limits.

```bash
docker compose up -d --build
```

### Verification
To ensure the container is running and healthy, you can check the logs or hit the health endpoint:
```bash
docker ps
curl http://localhost:8080/health
```

> [!TIP]
> The API and frontend are now securely running on port `8080`. Make sure to configure your AWS Security Group to allow inbound TCP traffic on port `8080` if you plan to access the frontend externally.
