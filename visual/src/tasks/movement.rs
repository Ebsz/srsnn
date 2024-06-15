use crate::tasks::TaskRenderer;

use tasks::movement_task::{MovementTask, AGENT_RADIUS, TARGET_RADIUS, SENSOR_LEN, WORLD_SIZE};

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;


impl TaskRenderer for MovementTask {
    fn render(&self, canvas: &mut WindowCanvas) {
        let _ = canvas.filled_circle(self.agent.x as i16, self.agent.y as i16, AGENT_RADIUS as i16, Color::BLACK);

        let _ = canvas.filled_circle(self.target.x as i16, self.target.y as i16, TARGET_RADIUS as i16, Color::RED);


        let x = self.agent.x + self.agent.rotation.cos() * SENSOR_LEN;
        let y = self.agent.y - self.agent.rotation.sin() * SENSOR_LEN;

        let _ = canvas.thick_line(self.agent.x as i16, self.agent.y as i16, x as i16, y as i16, 3, Color::BLACK);
    }

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32) {
        (WORLD_SIZE.0 as i32, WORLD_SIZE.1 as i32)
    }
}
