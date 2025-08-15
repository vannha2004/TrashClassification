# controller/robot_controller.py

# Tọa độ các vị trí (x, y) trên canvas robot
# Các giá trị này bạn chỉnh theo layout thật
BIN_COORDS = {
    "plastic":  (200, 500),   # Thùng nhựa
    "paper":    (400, 500),   # Thùng giấy
    "metal":    (600, 500),   # Thùng kim loại
    "organic":  (800, 500),   # Thùng hữu cơ
}

# Tọa độ giữa camera (điểm để gắp rác ban đầu)
CAMERA_CENTER = (0, 600)  # Bạn chỉnh lại theo vị trí thật trên canvas

def move_robot_to_bin(trash_type):
    """
    Trả về mode di chuyển và tọa độ mục tiêu.
    mode:
      - 'pickup' : di chuyển đến camera center
      - 'drop'   : di chuyển đến thùng
    """
    trash_type = trash_type.lower()

    if trash_type not in BIN_COORDS:
        return "idle", None

    # B1: đến giữa camera để gắp
    pickup_coords = CAMERA_CENTER

    # B2: đến thùng tương ứng
    bin_coords = BIN_COORDS[trash_type]

    # Trả về 2 giai đoạn
    return "pickup_and_drop", {
        "pickup": pickup_coords,
        "drop": bin_coords
    }
