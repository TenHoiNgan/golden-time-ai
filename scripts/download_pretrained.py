"""
Script kiểm tra và tải trước trọng số mô hình MobileNetV3 vào local cache của Torchvision.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def main():
    print("Đang kiểm tra và tải weights MobileNetV3-Small...")
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        print("Tải thành công! Mô hình đã sẵn sàng cho suy luận ngoại tuyến.")
    except Exception as e:
        print(f"Lỗi khi tải trọng số: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
