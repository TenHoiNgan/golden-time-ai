"""
Script tiện ích khởi chạy nhanh dịch vụ AI ở chế độ phát triển (Development mode).
"""

import os
import sys
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Khởi động Golden Time AI Service tại cổng {port}...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, app_dir=PROJECT_ROOT, reload=False)
