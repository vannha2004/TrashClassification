// js/config.js

// Kích thước của các khớp và tay gắp
export const JOINT_RADIUS_SHOULDER = 24;
export const JOINT_RADIUS_ELBOW = 22;
export const ARM_WIDTH = 50;

export const GRIPPER_MAIN_WIDTH = 20;
export const GRIPPER_MAIN_HEIGHT = 50;
export const GRIPPER_FINGER_LENGTH = 52;
export const GRIPPER_FINGER_THICKNESS = 16;

// Màu sắc và thông số cho giao diện cơ khí
export const COLORS = {
    ARM_BASE: '#B0B0B0',
    ARM_HIGHLIGHT: '#D0D0D0',
    JOINT_MAIN: '#222222',
    JOINT_BORDER: '#808080',
    GRIPPER_MAIN: '#555555',
    GRIPPER_ACCENT: '#999999',
    TARGET_POINT: '#FF00FF'
};

// Chiều dài mỗi arm
export const ARM_LENGTH = 550; // Giữ nguyên theo yêu cầu của bạn

// Giới hạn góc khuỷu tay (độ và radian)
export const MIN_ELBOW_ANGLE_DEG = 30;
export const MAX_ELBOW_ANGLE_DEG = 180;
export const MIN_ELBOW_ANGLE_RAD = (MIN_ELBOW_ANGLE_DEG * Math.PI) / 180;
export const MAX_ELBOW_ANGLE_RAD = (MAX_ELBOW_ANGLE_DEG * Math.PI) / 180;

// Tốc độ animation
export const ANIMATION_STEP = 15;