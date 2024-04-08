use plotters::prelude::*;

use ndarray::Array1;
use crate::record::{Record, RecordType, RecordDataType};

pub fn generate_plots(record: &Record) {

    // Potentials
    let single_pot = record.get_potentials().iter().map(|x| x[0]).collect();

    let plot_ok = plot_single_neuron_potential(single_pot);
    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }

    // Spikes
    let mut spikedata: Vec<Array1<f32>> = vec![];

    for i in record.get(RecordType::Spikes) {
        if let RecordDataType::Spikes(s) = i {
            spikedata.push(s.clone());
        } else {
            panic!("Error parsing spike records");
        }
    }

    let plot_ok = plot_spikes(spikedata);
    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}

pub fn plot_network_energy(energy: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    plot_single_variable(energy, "Energy", "Energy", "energy.png",&RED)
}

pub fn plot_single_neuron_potential(potentials: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    plot_single_variable(potentials, "Potential", "Potentials", "pots.png", &BLACK)
}

fn plot_single_variable(data: Vec<f32>, description: &str, caption: &str, filename: &str, color: &RGBColor) -> Result<(), Box<dyn std::error::Error>> {
    let max_x: f32 = data.len() as f32;
    let min_x: f32 = 0.0;

    let max_y: f32 = data.iter().fold(0.0f32, |acc, &x| if x > acc {x} else {acc});
    let min_y: f32 = data.iter().fold(0.0f32, |acc, &x| if x < acc {x} else {acc});

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 50).into_font())
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    let series = LineSeries::new(
        data.iter().enumerate().map(|(i, x)| (i as f32, *x)),
        color,
    );

    chart
        .draw_series(series)?
        .label(description)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));


    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    log::info!("Plot saved to {}", filename);

    Ok(())
}

pub fn plot_spikes(spikedata: Vec<Array1<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    let filename = "spikeplot.png";

    let max_x = spikedata.len() as i32;
    let max_y = spikedata[0].shape()[0] as i32;

    let points: Vec<(i32, i32)> = to_spike_points(&spikedata);

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Spike plot", ("sans-serif", 30).into_font())
        .margin(15)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0..max_x, 0..max_y)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        points.iter().map(|p| Circle::new(*p, 3, BLACK.filled())),
    ).unwrap();

    root.present()?;

    log::info!("Plot saved to {}", filename);

    Ok(())
}

/// Convert the raw record data to points that can be plotted
fn to_spike_points(spikedata: &Vec<Array1<f32>>) -> Vec<(i32, i32)> {
    let mut points: Vec<(i32, i32)> = vec![];

    for (t, s) in spikedata.iter().enumerate() {
        // Get indices i of neurons that fired at time t
        // TODO: This is duplicated in FiringState; use
        let ixs: Vec<usize> = s.iter().enumerate().filter(|(_, n)| **n != 0.0).map(|(i,_)| i).collect();

        // Map from i to (t, i)
        let mut k: Vec<(i32,i32)> = ixs.iter().map(|i| (t as i32, *i as i32)).collect();

        points.append(&mut k);
    }

    points
}
