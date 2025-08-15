// js/animation.js
import { drawRobot } from './robot.js';

export function animate(ctx, canvas) {
    drawRobot(ctx, canvas);
    requestAnimationFrame(() => animate(ctx, canvas));
}