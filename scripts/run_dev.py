"""
Script tiện ích khởi chạy nhanh dịch vụ AI ở chế độ phát triển (Development mode).
"""

import sys
import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Khởi động Golden Time AI Service tại cổng {port}...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
