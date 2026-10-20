import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FruitClass(BaseModel):
    """Định nghĩa metadata của một lớp trái cây trong hệ sinh thái Golden Time."""

    class_id: str = Field(..., description="Mã định danh duy nhất của lớp (snake_case, không dấu)")
    display_name: str = Field(..., description="Tên tiếng Việt hiển thị chính thức của loại quả")
    folder_name: str = Field(..., description="Tên thư mục con trong dataset")
    aliases: List[str] = Field(default_factory=list, description="Các tên gọi thay thế, tên tiếng Anh hoặc biến thể")
    profile: Dict[str, Any] = Field(default_factory=dict, description="Thông tin đặc trưng phục vụ heuristic và gợi ý")


class FruitRegistry:
    """
    Quản lý tập trung toàn bộ danh mục lớp trái cây (Single Source of Truth).
    Đảm bảo tính nhất quán giữa Dataset, Training, Model Artifacts và API Inference.
    """

    def __init__(self, config_path: Optional[str] = None):
        self._classes: List[FruitClass] = []
        self._by_id: Dict[str, FruitClass] = {}
        self._by_folder: Dict[str, FruitClass] = {}
        self._by_display_name: Dict[str, FruitClass] = {}
        self._by_alias: Dict[str, FruitClass] = {}
        self._config_path: str = ""

        if config_path:
            self.load(config_path)
        else:
            default_path = os.path.join(os.path.dirname(__file__), "classes.json")
            if os.path.exists(default_path):
                self.load(default_path)

    def load(self, file_path: str) -> None:
        """Tải và xác thực danh mục lớp từ file JSON."""
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Không tìm thấy file cấu hình lớp trái cây tại: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Dữ liệu trong {abs_path} phải là danh sách (JSON array).")

        classes: List[FruitClass] = []
        seen_ids: Set[str] = set()
        seen_folders: Set[str] = set()
        seen_names: Set[str] = set()

        for item in data:
            fc = FruitClass(**item)
            if fc.class_id in seen_ids:
                raise ValueError(f"Trùng lặp class_id: '{fc.class_id}' trong {abs_path}")
            if fc.folder_name in seen_folders:
                raise ValueError(f"Trùng lặp folder_name: '{fc.folder_name}' trong {abs_path}")
            if fc.display_name in seen_names:
                raise ValueError(f"Trùng lặp display_name: '{fc.display_name}' trong {abs_path}")

            seen_ids.add(fc.class_id)
            seen_folders.add(fc.folder_name)
            seen_names.add(fc.display_name)
            classes.append(fc)

        self._classes = classes
        self._by_id = {c.class_id: c for c in classes}
        self._by_folder = {c.folder_name: c for c in classes}
        self._by_display_name = {c.display_name: c for c in classes}
        self._by_alias = {}

        for c in classes:
            self._by_alias[c.class_id.lower()] = c
            self._by_alias[c.display_name.lower()] = c
            self._by_alias[c.folder_name.lower()] = c
            for alias in c.aliases:
                self._by_alias[alias.lower()] = c

        self._config_path = abs_path
        logger.info("Đã tải thành công %d lớp trái cây từ %s", len(self._classes), abs_path)

    @property
    def total_classes(self) -> int:
        return len(self._classes)

    def get_by_id(self, class_id: str) -> Optional[FruitClass]:
        return self._by_id.get(class_id)

    def get_by_folder(self, folder_name: str) -> Optional[FruitClass]:
        return self._by_folder.get(folder_name)

    def get_by_name(self, display_name: str) -> Optional[FruitClass]:
        return self._by_display_name.get(display_name)

    def match_any(self, query: str) -> Optional[FruitClass]:
        """Tìm kiếm linh hoạt theo class_id, display_name, folder_name hoặc alias."""
        if not query:
            return None
        cleaned = query.strip().lower()
        return self._by_alias.get(cleaned)

    def get_display_names(self) -> List[str]:
        return [c.display_name for c in self._classes]

    def get_class_ids(self) -> List[str]:
        return [c.class_id for c in self._classes]

    def get_folder_names(self) -> List[str]:
        return [c.folder_name for c in self._classes]

    def validate_dataset_structure(self, split_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Kiểm tra cấu trúc thư mục dataset đối chiếu với registry.
        Trả về: (matched_folders, missing_folders, unexpected_folders)
        """
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Thư mục không tồn tại: {split_dir}")

        existing = [e.name for e in os.scandir(split_dir) if e.is_dir()]
        expected = set(self._by_folder.keys())
        existing_set = set(existing)

        matched = sorted(list(expected.intersection(existing_set)))
        missing = sorted(list(expected.difference(existing_set)))
        unexpected = sorted(list(existing_set.difference(expected)))

        return matched, missing, unexpected


# Singleton toàn cục
fruit_registry = FruitRegistry()
