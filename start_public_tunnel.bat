@echo off
title Bovista Global Public Tunnel
echo Starting Bovista Public Tunnel via Cloudflare...
.\cloudflared.exe tunnel --url http://127.0.0.1:5000
pause
