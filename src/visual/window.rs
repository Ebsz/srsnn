use sdl2::Sdl;
use sdl2::event::Event;
use sdl2::render::WindowCanvas;

use sdl2::rect::Rect;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;

use sdl2::gfx::primitives::DrawRenderer;

use std::time::Instant;

use crate::task_executor::{TaskExecutor, ExecutionState};
use tasks::cognitive_task::{CognitiveTask, TaskRenderer};


const TARGET_FPS: u128 = 60;

pub struct TaskWindow<'a, T: CognitiveTask + TaskRenderer> {
    executor: TaskExecutor<'a, T>,
    canvas: WindowCanvas,
    context: Sdl
}

impl<'a, T: CognitiveTask + TaskRenderer> TaskWindow<'a, T> {
    pub fn new(executor: TaskExecutor<T>, size: (u32, u32)) -> TaskWindow<T> {
        let sdl_context = sdl2::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();

        let window = video_subsystem.window("window", size.0,size.1)
            .position_centered()
            .build()
            .unwrap();

        let canvas = window.into_canvas().build().unwrap();

        TaskWindow {
            context: sdl_context,
            canvas,
            executor,
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
                    Event::KeyDown {keycode: Some(Keycode::R), ..} => {
                        self.executor.reset()

                    }
                    _ => {}
                }
            }

            self.render();
            self.canvas.present();
        }
    }

    fn update(&mut self) {
        if self.executor.state != ExecutionState::FINISHED {
            self.executor.step(false);
        }
    }

    fn render(&mut self) {
        self.canvas.set_draw_color(Color::RGB(255, 255, 255));
        self.canvas.clear();

        self.executor.task.render(&mut self.canvas);
    }
}
