/// Record of the time-series evolution of a spiking network.
///
/// All data points are real vectors, stored as Array1<f32>.

use ndarray::Array1;

use std::collections::HashMap;


#[derive(Clone)]
pub struct Record {
    pub records: HashMap<RecordType, Vec<Array1<f32>>>
}

impl Record {
    pub fn new() -> Record {
        let mut records = HashMap::new();

        records.insert(RecordType::Spikes, Vec::new());
        records.insert(RecordType::Potentials, Vec::new());
        records.insert(RecordType::SynapticCurrent, Vec::new());
        records.insert(RecordType::InputSpikes, Vec::new());
        records.insert(RecordType::OutputSpikes, Vec::new());

        Record {
            records
        }
    }

    pub fn log(&mut self, record_type: RecordType, data: Array1<f32>) {
        if let Some(d) = self.records.get_mut(&record_type) {
            d.push(data);
        }
    }

    pub fn get(&self, record_type: RecordType) -> Vec<Array1<f32>> {
        self.get_ref(record_type).iter().map(|r| (*r).clone()).collect()
    }

    pub fn get_ref(&self, record_type: RecordType) -> &[Array1<f32>] {
        if let Some(d) = self.records.get(&record_type) {
            d
        } else {
            panic!("Could not get record of type {:?}", record_type);
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum RecordType {
    Spikes,
    Potentials,
    InputSpikes,
    OutputSpikes,
    SynapticCurrent,
}
