pub mod movement;
pub mod catching;
pub mod survival;
pub mod pole_balancing;

use sdl2::render::WindowCanvas;

/// Defines the visual representation of a task.
pub trait TaskRenderer {
    fn render(&self, canvas: &mut WindowCanvas);

    fn render_size() -> (i32, i32);
}
