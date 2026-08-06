import sys, os
sys.path.insert(0, 'src')
os.environ['PYTHONPATH'] = 'src'

from litert_studio.server.app import create_app
from starlette.testclient import TestClient

app = create_app()
client = TestClient(app)

response = client.get('/api/system')
print(f'Status: {response.status_code}')
data = response.json()
print(f'Platform: {data.get("platform")}')
print(f'Accelerator: {data.get("accelerator")}')
print(f'Has packages: {"packages" in data}')
if response.status_code == 200:
    print('SUCCESS: /api/system returns 200 with graceful degradation')
else:
    print(f'FAIL: /api/system returned {response.status_code}')
    print(response.text)
