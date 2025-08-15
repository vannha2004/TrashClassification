// js/event-listeners.js
import { setTargetGripperPosition, centerX, centerY } from './robot.js';

export function setupEventListeners(canvas, mainContainer, robotPanel) {
    mainContainer.addEventListener('click', (event) => {
        const rect = robotPanel.getBoundingClientRect(); // dùng canvas hoặc robotPanel
        let mouseX = event.clientX - rect.left;
        let mouseY = event.clientY - rect.top;

        const currentBaseRotationRad = 0;

        const newTargetX =
            (mouseX - centerX) * Math.cos(-currentBaseRotationRad) -
            (mouseY - centerY) * Math.sin(-currentBaseRotationRad);

        const newTargetY =
            (mouseX - centerX) * Math.sin(-currentBaseRotationRad) +
            (mouseY - centerY) * Math.cos(-currentBaseRotationRad);

        setTargetGripperPosition(newTargetX, newTargetY);
    });

}