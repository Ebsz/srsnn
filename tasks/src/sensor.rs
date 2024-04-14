pub struct Sensor {
    pub length: f32,
    pub angle: f32
}

impl Sensor {
    pub fn new(length: f32, angle: f32) -> Sensor {
        Sensor {
            length,
            angle
        }
    }

    /// Calculates the endpoint of the sensor based on its starting point and angle
    pub fn endpoint(&self, pos: (f32,f32)) -> (f32, f32) {
        let x = pos.0 + self.angle.cos() * self.length;
        let y = pos.1 - self.angle.sin() * self.length;

        (x, y)
    }

    /// Returns a number k in the range (0, 1) if the sensor intersects with the target circle,
    /// where k=0.0 if the sensor is at is max length, k=1.0 if it is at its shortest.
    ///
    /// Returns 0.0 if there is no intersection
    pub fn read(&self, pos: (f32,f32), target_pos: (f32, f32), target_radius: f32) -> f32 {
        let h = target_pos.0;
        let k = target_pos.1;

        let x0 = pos.0;
        let y0 = pos.1;

        // end points of the line
        let x1 = x0 + self.angle.cos() * self.length;
        let y1 = y0 - self.angle.sin() * self.length;

        let a = (x1 - x0).powf(2.0) + (y1 - y0).powf(2.0);
        let b = 2.0 * (x1 -x0) * (x0 - h) + 2.0 * (y1 - y0) * (y0 - k);
        let c = (x0 - h).powf(2.0) + (y0 - k).powf(2.0) - target_radius.powf(2.0);

        let di = b.powf(2.0) - 4.0 * a * c;

        if di > 0.0 {
            let t = 2.0 * c / (-b + di.sqrt());

            if t >= 0.0 && t <= 1.0 {
                return 1.0 - t;
            }
        }

        0.0
    }
}

