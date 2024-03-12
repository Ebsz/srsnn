use tasks::cognitive_task::CognitiveTask;
use tasks::catching_task::CatchingTask;

use macroquad::prelude::*;


#[macroquad::main("Apple")]
async fn main() {

    let mut task = CatchingTask::new();

    println!("{:?}", (task.agent.x, task.agent.y));

    request_new_screen_size(800.0,600.0);
    set_fullscreen(false);

    let mut finished: bool = false;

    loop {
        if !finished {
            let res = task.tick();

            if let Some(r) = res {
                println!("{:?}", r);
                finished = true;
            }
        }

        if is_key_down(KeyCode::A) {
            task.agent.move_left();
        }
        if is_key_down(KeyCode::D) {
            task.agent.move_right();
        }

        clear_background(LIGHTGRAY);

        draw_rectangle(0.0,0.0, 800.0, 600.0, WHITE);

        // Draw apple
        draw_circle(task.apple.x as f32,
            task.apple.y as f32,
            task.apple.r as f32,
            RED);

        // Draw Sensors
        let agent_pos = (task.agent.x as f32, task.agent.y as f32);

        for s in &task.sensors {
            //let x = pos.0 + s.angle.cos() * SENSOR_LEN;
            //let y = pos.1 - s.angle.sin() * SENSOR_LEN;

            let sensor_endpoint = s.endpoint(agent_pos);

            draw_line(agent_pos.0, agent_pos.1, sensor_endpoint.0, sensor_endpoint.1, 2.0, BLACK);

        }

        // Draw agent
        draw_circle(task.agent.x as f32,
            task.agent.y as f32,
            task.agent.r as f32,
            BLUE);



        next_frame().await
    }
}
