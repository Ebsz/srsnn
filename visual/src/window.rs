use sdl2::Sdl;
use sdl2::event::Event;
use sdl2::render::WindowCanvas;

use sdl2::gfx::primitives::DrawRenderer;

use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;

use std::time::Instant;

const TARGET_FPS: u128 = 60;

const CLEAR_COLOR: Color = Color::RGB(60, 65, 70);


pub struct Window {
    components: Vec<Box<dyn WindowComponent>>,

    canvas: WindowCanvas,
    context: Sdl,

    size: (u32, u32)
}

impl Window {
    pub const DEFAULT_WINDOW_SIZE: (u32, u32) = (1200, 800);

    pub fn new(size: (u32, u32), components: Vec<Box<dyn WindowComponent>>) -> Window {
        let sdl_context = sdl2::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();

        let window = video_subsystem.window("window", size.0, size.1)
            .position_centered()
            .build()
            .unwrap();

        let canvas = window.into_canvas().build().unwrap();

        Window {
            components: components,
            canvas,
            context: sdl_context,
            size
        }
    }

    pub fn run(&mut self) {
        self.canvas.clear();
        self.canvas.present();

        let mut event_pump = self.context.event_pump().unwrap();

        let mut now = Instant::now();
        let mut last = now;
        let mut delta;

        let mut acc: u128 = 0;
        let mut ticks = 0;

        let mut print_acc: u128 = 0;

        let mut running: bool = true;
        while running {
            now = Instant::now();
            delta = now-last;

            last = now;
            acc += delta.as_nanos();
            print_acc += delta.as_nanos();

            if acc > 1_000_000_000 / TARGET_FPS {
                self.update();

                ticks += 1;
                acc -= 1_000_000_000 / TARGET_FPS;
            }

            if print_acc >= 1_000_000_000 {
                println!("ticks: {ticks}");
                ticks = 0;
                print_acc = 0;
            }

            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit {..} |
                    Event::KeyDown {keycode: Some(Keycode::Escape), ..} => {
                        running = false;
                    },
                    _ => {}
                }
            }

            self.render();
            self.canvas.present();
        }
    }

    fn update(&mut self) {
        for c in &mut self.components {
            c.update();
        }
    }

    fn render(&mut self) {
        self.canvas.set_draw_color(CLEAR_COLOR);
        self.canvas.clear();

        for c in &self.components {
            c.render( &mut DrawContext { size: self.size, canvas: &self.canvas });
        }
    }
}

// TODO: Draw on surfaces instead and put them together
pub struct DrawContext<'a> {
    pub size: (u32, u32),
    pub canvas: &'a WindowCanvas
}

impl DrawContext<'_> {
    pub fn draw_circle(&mut self, x: i16, y: i16, r: i16, color: Color) {
        let _ = self.canvas.aa_circle(x, y, r, color);
        let _ = self.canvas.filled_circle(x, y, r, color);
    }

    pub fn draw_line(&mut self, p1: (i16, i16), p2: (i16, i16), width: u8, color: Color) {
        let _ = self.canvas.thick_line(p1.0, p1.1, p2.0, p2.1, width, color);
    }
}

pub trait WindowComponent {
    fn update(&mut self);
    fn render(&self, context: &mut DrawContext);
}
