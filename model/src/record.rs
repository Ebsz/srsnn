/// Record of the time-series evolution of a spiking network.
///
/// All data points are real vectors, stored as Array1<f32>.

use ndarray::{Array, Array1, Array2};

use std::collections::HashMap;


pub type RecordData = Array1<f32>;

#[derive(Clone)]
pub struct Record {
    pub records: HashMap<RecordType, Vec<RecordData>>
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

    pub fn log(&mut self, record_type: RecordType, data: RecordData) {
        if let Some(d) = self.records.get_mut(&record_type) {
            d.push(data);
        }
    }

    pub fn get(&self, record_type: RecordType) -> Vec<RecordData> {
        self.get_ref(record_type).iter().map(|r| (*r).clone()).collect()
    }

    pub fn get_ref(&self, record_type: RecordType) -> &[RecordData] {
        if let Some(d) = self.records.get(&record_type) {
            d
        } else {
            panic!("Could not get record of type {:?}", record_type);
        }
    }

    pub fn as_array(&self, record_type: RecordType) -> Array2<f32>{
        let record = self.get_ref(record_type);

        let mut out: Array2<f32> = Array::zeros((0, record[0].shape()[0]));

        for r in record {
            out.push_row(r.into());
        }

        out
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
