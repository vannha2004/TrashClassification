// js/robot.js
import {
  JOINT_RADIUS_SHOULDER,
  JOINT_RADIUS_ELBOW,
  ARM_WIDTH,
  GRIPPER_MAIN_WIDTH,
  GRIPPER_MAIN_HEIGHT,
  GRIPPER_FINGER_LENGTH,
  GRIPPER_FINGER_THICKNESS,
  COLORS,
  ARM_LENGTH,
  MIN_ELBOW_ANGLE_RAD,
  MAX_ELBOW_ANGLE_RAD,
  ANIMATION_STEP,
} from "./config.js";

// Biến trạng thái của robot (sẽ được cập nhật từ main.js)
export let targetGripperX = 0;
export let targetGripperY = ARM_LENGTH * 2; // Vị trí ban đầu
export let currentGripperX = targetGripperX;
export let currentGripperY = targetGripperY;

export let centerX; // Sẽ được thiết lập trong setupCanvasSize
export let centerY; // Sẽ được thiết lập trong setupCanvasSize

export let gripperOpen = GRIPPER_FINGER_LENGTH; // trạng thái hiện tại của ngón
export let targetGripperOpen = GRIPPER_FINGER_LENGTH; // trạng thái mục tiêu
export const GRIPPER_SPEED = 2; // tốc độ animation mỗi frame

// Hàm để cập nhật vị trí đích của tay gắp
export function setTargetGripperPosition(x, y) {
  targetGripperX = x;
  targetGripperY = y;
}

// Hàm để thiết lập vị trí gốc của robot (center X, Y trên canvas)
export function setRobotCenter(x, y) {
  centerX = x;
  centerY = y;
}

export function grab() {
  targetGripperOpen = 15; // ví dụ: rút ngón lại gần thân
}

export function release() {
  targetGripperOpen = GRIPPER_FINGER_LENGTH; // trở về chiều dài ban đầu
}

// Hàm vẽ robot
export function drawRobot(ctx, canvas) {
  // --- Cập nhật trạng thái mở/đóng của ngón ---
  if (gripperOpen < targetGripperOpen) {
    gripperOpen += GRIPPER_SPEED;
    if (gripperOpen > targetGripperOpen) gripperOpen = targetGripperOpen;
  } else if (gripperOpen > targetGripperOpen) {
    gripperOpen -= GRIPPER_SPEED;
    if (gripperOpen < targetGripperOpen) gripperOpen = targetGripperOpen;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();

  // Di chuyển gốc tọa độ về centerX, centerY
  ctx.translate(centerX, centerY);

  // --- Cập nhật vị trí hiện tại theo kiểu tuyến tính ---
  const dx = targetGripperX - currentGripperX;
  const dy = targetGripperY - currentGripperY;
  const distanceToTarget = Math.sqrt(dx * dx + dy * dy);

  if (distanceToTarget > ANIMATION_STEP) {
    currentGripperX += (dx / distanceToTarget) * ANIMATION_STEP;
    currentGripperY += (dy / distanceToTarget) * ANIMATION_STEP;
  } else {
    currentGripperX = targetGripperX;
    currentGripperY = targetGripperY;
  }

  // --- Kiểm tra tầm với của robot ---
  const distance = Math.sqrt(
    currentGripperX * currentGripperX + currentGripperY * currentGripperY
  );
  const maxReach = 2 * ARM_LENGTH;
  const minReach = Math.sqrt(
    ARM_LENGTH * ARM_LENGTH +
      ARM_LENGTH * ARM_LENGTH -
      2 * ARM_LENGTH * ARM_LENGTH * Math.cos(MIN_ELBOW_ANGLE_RAD)
  );

  if (distance > maxReach) {
    const angleToCurrent = Math.atan2(currentGripperY, currentGripperX);
    currentGripperX = Math.cos(angleToCurrent) * maxReach;
    currentGripperY = Math.sin(angleToCurrent) * maxReach;
  } else if (distance < minReach) {
    const angleToCurrent = Math.atan2(currentGripperY, currentGripperX);
    currentGripperX = Math.cos(angleToCurrent) * minReach;
    currentGripperY = Math.sin(angleToCurrent) * minReach;
  }

  // --- GIẢI QUYẾT BÀI TOÁN INVERSE KINEMATICS (IK) ---
  let elbowAngleRad = Math.acos(
    (ARM_LENGTH * ARM_LENGTH + ARM_LENGTH * ARM_LENGTH - distance * distance) /
      (2 * ARM_LENGTH * ARM_LENGTH)
  );
  elbowAngleRad = Math.max(
    MIN_ELBOW_ANGLE_RAD,
    Math.min(MAX_ELBOW_ANGLE_RAD, elbowAngleRad)
  );

  const angleToTarget = Math.atan2(currentGripperY, currentGripperX);
  let alpha = Math.acos(
    (ARM_LENGTH * ARM_LENGTH + distance * distance - ARM_LENGTH * ARM_LENGTH) /
      (2 * ARM_LENGTH * distance)
  );
  let shoulderAngleRad = angleToTarget - alpha;

  const elbowX = ARM_LENGTH * Math.cos(shoulderAngleRad);
  const elbowY = ARM_LENGTH * Math.sin(shoulderAngleRad);

  const angleArm2 = shoulderAngleRad + (Math.PI - elbowAngleRad);
  const gripperX = elbowX + ARM_LENGTH * Math.cos(angleArm2);
  const gripperY = elbowY + ARM_LENGTH * Math.sin(angleArm2);

  // --- Bắt đầu vẽ robot với giao diện mới ---

  ctx.shadowBlur = 8;
  ctx.shadowOffsetX = 3;
  ctx.shadowOffsetY = 3;
  ctx.shadowColor = "rgba(0,0,0,0.25)";
  // --- Vẽ Arm 1 (Hình chữ nhật với gradient và viền) ---
  ctx.save();
  ctx.rotate(shoulderAngleRad);

  const arm1Gradient = ctx.createLinearGradient(
    0,
    -ARM_WIDTH / 2,
    0,
    ARM_WIDTH / 2
  );
  arm1Gradient.addColorStop(0, COLORS.ARM_HIGHLIGHT);
  arm1Gradient.addColorStop(0.5, COLORS.ARM_BASE);
  arm1Gradient.addColorStop(1, COLORS.ARM_HIGHLIGHT);

  ctx.fillStyle = arm1Gradient;
  ctx.fillRect(0, -ARM_WIDTH / 2, ARM_LENGTH, ARM_WIDTH);

  ctx.strokeStyle = COLORS.JOINT_BORDER;
  ctx.lineWidth = 2;
  ctx.strokeRect(0, -ARM_WIDTH / 2, ARM_LENGTH, ARM_WIDTH);

  ctx.restore();

  // --- Vẽ khớp vai (gốc - hình tròn màu đen nổi bật với viền) ---
  ctx.beginPath();
  ctx.arc(0, 0, JOINT_RADIUS_SHOULDER, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.JOINT_MAIN;
  ctx.fill();
  ctx.strokeStyle = COLORS.JOINT_BORDER;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.closePath();

  //
  ctx.save();
  ctx.translate(gripperX, gripperY);
  ctx.rotate(angleArm2);

  ctx.save();
  ctx.translate(gripperX, gripperY);
  ctx.rotate(angleArm2);
  ctx.restore();

  // Ngón trên
  ctx.save();
  ctx.translate(open, 0);
  ctx.rotate((Math.PI * 3) / 4);
  ctx.fillRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.strokeRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.restore();

  // Ngón dưới
  ctx.save();
  ctx.translate(-open, 0);
  ctx.rotate(Math.PI / 4);
  ctx.fillRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.strokeRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.restore();

  // Ngón trái (ngửa ngang)
  ctx.save();
  ctx.translate(-open, 0);
  ctx.rotate(-Math.PI / 4);
  ctx.fillRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.strokeRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.restore();

  // Ngón phải (ngửa ngang)
  ctx.save();
  ctx.translate(open, 0);
  ctx.rotate((-Math.PI * 3) / 4);
  ctx.fillRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.strokeRect(
    -GRIPPER_FINGER_THICKNESS / 2,
    0,
    GRIPPER_FINGER_THICKNESS,
    gripperOpen
  );
  ctx.restore();

  ctx.restore();

  // --- Vẽ Arm 2 (Hình chữ nhật với gradient và viền) ---
  ctx.save();
  ctx.translate(elbowX, elbowY);
  ctx.rotate(angleArm2);

  const arm2Gradient = ctx.createLinearGradient(
    0,
    -ARM_WIDTH / 2,
    0,
    ARM_WIDTH / 2
  );
  arm2Gradient.addColorStop(0, COLORS.ARM_HIGHLIGHT);
  arm2Gradient.addColorStop(0.5, COLORS.ARM_BASE);
  arm2Gradient.addColorStop(1, COLORS.ARM_HIGHLIGHT);
  ctx.fillStyle = arm2Gradient;

  ctx.fillRect(0, -ARM_WIDTH / 2, ARM_LENGTH, ARM_WIDTH);

  ctx.strokeStyle = COLORS.JOINT_BORDER;
  ctx.lineWidth = 2;
  ctx.strokeRect(0, -ARM_WIDTH / 2, ARM_LENGTH, ARM_WIDTH);

  ctx.restore();

  // --- Vẽ khớp khuỷu tay (hình tròn màu đen nổi bật với viền) ---
  ctx.beginPath();
  ctx.arc(elbowX, elbowY, JOINT_RADIUS_ELBOW, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.JOINT_MAIN;
  ctx.fill();
  ctx.strokeStyle = COLORS.JOINT_BORDER;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.closePath();

  // --- Vẽ tay gắp (kiểu kẹp đơn giản) ---
  ctx.save();
  ctx.translate(gripperX, gripperY);
  ctx.rotate(angleArm2);

  // Vẽ phần khớp chính của tay gắp
  ctx.beginPath();
  ctx.arc(0, 0, JOINT_RADIUS_SHOULDER, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.JOINT_MAIN;
  ctx.fill();
  ctx.strokeStyle = COLORS.JOINT_BORDER;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.closePath();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(targetGripperX, targetGripperY, 5, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.TARGET_POINT;
  ctx.fill();
  ctx.closePath();

  ctx.setLineDash([5, 5]); // đường nét đứt
  ctx.strokeStyle = COLORS.TARGET_POINT; // màu target
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0); // vai là gốc
  ctx.lineTo(targetGripperX, targetGripperY); // đến vị trí mục tiêu
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.restore();
}
