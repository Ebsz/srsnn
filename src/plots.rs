use plotters::prelude::*;

pub fn plot_single_neuron_potential(potentials: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let max_x: f32 = potentials.len() as f32;
    let min_x: f32 = 0.0;

    let max_y: f32 = 3.0;
    let min_y: f32 = -100.0;

    let filename = "pots.png";

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Potentials", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    let pots_series = LineSeries::new(
        potentials.iter().enumerate().map(|(i, x)| (i as f32, *x)),
        &BLACK,
    );

    chart
        .draw_series(pots_series)?
        .label("Potential")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));


    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Plot saved to {}", filename);

    Ok(())
}
