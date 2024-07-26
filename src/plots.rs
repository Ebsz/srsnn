use plotters::prelude::*;

use ndarray::{s, Array, Array1, Array2, Array3};
use model::record::{Record, RecordType, RecordDataType};

use evolution::stats::EvolutionStatistics;


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

    let plot_ok = plot_spikes(spikedata.clone());

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }

    let mut spike_array: Array2<f32> = Array::zeros((spikedata.len(), spikedata[0].shape()[0]));

    for (i, p) in spikedata.iter().enumerate() {
        //println!("{:?}, {:?}", i, p);
        spike_array.slice_mut(s!(i,..)).assign(&p);
    }

    //let psth = crate::analysis::to_firing_rate(spike_array);
    //plot_firing_rates(psth);
}

pub fn plot_all_potentials(record: &Record) {
    let mut pots: Vec<Vec<f32>> = Vec::new();
    let potentials = record.get_potentials();

    for i in 0..potentials[0].shape()[0] {
        pots.push(potentials.iter().map(|x| x[i]).collect());
    }

    let plot_ok = plt::plot_multiple_series(pots, "Potential", "Potentials", "allpots.png");

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}

pub fn plot_evolution_stats(stats: &EvolutionStatistics) {
    log::info!("Plotting evolution stats");

    let _ = plt::plot_single_variable(stats.generation_best_fitness.clone(), "Generation best", "Evolution", "evolution_best.png", &BLUE);
    let _ = plt::plot_single_variable(stats.generation_mean_fitness.clone(), "Generation mean", "Evolution", "evolution_mean.png", &BLUE);
}

pub fn plot_network_energy(energy: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    plt::plot_single_variable(energy, "Energy", "Energy", "energy.png", &RED)
}

pub fn plot_single_neuron_potential(potentials: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    plt::plot_single_variable(potentials, "Potential", "Potentials", "pots.png", &BLACK)
}

pub fn plot_firing_rates(data: Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let filename = "firing_rates.png";
    //let points = data.iter().map()

    let max_x = data.nrows();
    let max_y = data.ncols();

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Network firing rates", ("sans-serif", 30).into_font())
        .margin(15)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0..max_x, 0..max_y)?;


    chart.configure_mesh().draw()?;

    //chart.draw_series(
    //points.iter().map(|p|

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

pub fn plot_degree_distribution(data: &Array1<u32>) {
    let _ = plt::histogram(data, "dgdist.png", "Degree distribution");
}


pub mod plt {
    use super::*;
    use utils::math;

    pub fn plot_single_variable(
        data: Vec<f32>,
        description: &str,
        caption: &str,
        filename: &str,
        color: &RGBColor)
        -> Result<(), Box<dyn std::error::Error>> {

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

    /// Plot multiple series
    pub fn plot_multiple_series(
        data: Vec<Vec<f32>>,
        description: &str,
        caption: &str,
        filename: &str)
        -> Result<(), Box<dyn std::error::Error>> {
        let max_x: f32 = data[0].len() as f32;
        let min_x: f32 = 0.0;

        let max_y: f32 = data[0].iter().fold(0.0f32, |acc, &x| if x > acc {x} else {acc});
        let min_y: f32 = data[0].iter().fold(0.0f32, |acc, &x| if x < acc {x} else {acc});

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 50).into_font())
            .margin(15)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        //let colors: Vec<RGBColor> = (0..data.len()).map(|i| RGBColor((0 + i * 20) as u8, 0 , (255 - i * 20) as u8)).collect();

        for d in data.iter() {
            let series = LineSeries::new(
                d.iter().enumerate().map(|(i, x)| (i as f32, *x)),
                &RED,
            );

            chart
                .draw_series(series)?
                .label(description)
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    /// Plot two variables against each other
    pub fn plot_two(
        points: Vec<(f32, f32)>,
        filename: &str,
        caption: &str)
        -> Result<(), Box<dyn std::error::Error>> {

        let xs: Vec<f32> = points.iter().map(|(x,_)| *x).collect();
        let ys: Vec<f32> = points.iter().map(|(_,y)| *y).collect();

        let max_x: f32 = math::maxf(&xs);
        let max_y: f32 = math::maxf(&ys);
        let min_x: f32 = math::minf(&xs);
        let min_y: f32 = math::minf(&ys);

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 30).into_font())
            .margin(15)
            .x_label_area_size(20)
            .y_label_area_size(20)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        let series = LineSeries::new(points, &BLACK);
        chart.draw_series(series).unwrap();

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    pub fn histogram(
        data: &Array1<u32>,
        filename: &str,
        caption: &str)
        -> Result<(), Box<dyn std::error::Error>> {

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_x: u32 = data.shape()[0] as u32 / 2;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(35)
            .build_cartesian_2d((0u32..max_x).into_segmented(), 0u32..10u32)?;

        chart.configure_mesh()
            .disable_x_mesh()
            .bold_line_style(WHITE.mix(0.3))
            .y_desc("Count")
            .x_desc("Bucket")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(RED.mix(0.5).filled())
                .data(data.iter().map(|x: &u32| (*x, 1)))
        )?;

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    pub fn plot_3d(
        data: &Array3<u32>,
        filename: &str,
        caption: &str)
        -> Result<(), Box<dyn std::error::Error>> {

        let min_x = 0.0;
        let min_y = 0.0;
        let max_z = 0.0;
        let max_x = 1.0;
        let max_y = 1.0;
        let max_z = 1.0;

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(35)
            .build_cartesian_3d(min_x..max_x, min_y..max_y, min_x..max_x)?;

        chart.configure_axes().draw()?;

        Ok(())
    }

    //pub fn plot_heatmap() -> Result<(), Box<dyn std::error::Error>> {
    //    let filename = "heatmap.png";
    //    let caption = "heatmap";
    //
    //    let max_x: f32 = 1000.0;
    //    let min_x: f32 = 0.0;
    //
    //    let max_y: f32 = 200.0;
    //    let min_y: f32 = 0.0;
    //
    //    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    //    root.fill(&WHITE)?;
    //
    //    let mut chart = ChartBuilder::on(&root)
    //        .caption(caption, ("sans-serif", 50).into_font())
    //        .margin(15)
    //        .x_label_area_size(30)
    //        .y_label_area_size(30)
    //        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;
    //
    //    chart.draw_series(
    //        points.iter().map(|p| Circle::new(*p, 3, BLACK.filled())),
    //    ).unwrap();
    //
    //    root.present()?;
}
