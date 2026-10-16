from pyngrok import ngrok
import time

t = ngrok.connect(5000, "http")
print('ngrok_public_url:', t.public_url)
print('Tunnel established. Press Ctrl+C to stop.')
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    ngrok.disconnect(t.public_url)
    print('Tunnel closed')
