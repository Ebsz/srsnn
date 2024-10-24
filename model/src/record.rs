use ndarray::Array1;

use std::collections::HashMap;


#[derive(Clone)]
pub struct Record {
    pub records: HashMap<RecordType, Vec<RecordDataType>>
}

impl Record {
    pub fn new() -> Record {
        let mut records = HashMap::new();

        records.insert(RecordType::Spikes, Vec::new());
        records.insert(RecordType::Potentials, Vec::new());
        records.insert(RecordType::InputSpikes, Vec::new());
        records.insert(RecordType::OutputSpikes, Vec::new());

        Record {
            records
        }
    }

    pub fn log(&mut self, record_type: RecordType, data: RecordDataType) {
        if let Some(d) = self.records.get_mut(&record_type) {
            d.push(data);
        }
    }

    pub fn get(&self, record_type: RecordType) -> &[RecordDataType] {
        if let Some(d) = self.records.get(&record_type) {
            d
        } else {
            panic!("Could not get record of type {:?}", record_type);
        }
    }

    pub fn get_record(&self, record_type: RecordType) -> Vec<Array1<f32>> {
        let rec = self.get(record_type);
        let mut data = vec![];

        for r in rec {
            if let RecordDataType::Potentials(p) = r {
                data.push(p.clone())
            } else {
                panic!("Could not get record!")
            }
        }

        data
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum RecordType {
    Spikes,
    Potentials,
    InputSpikes,
    OutputSpikes
}

#[derive(Clone, Debug)]
pub enum RecordDataType {
    Spikes(Array1<f32>),
    Potentials(Array1<f32>),
    InputSpikes(Array1<f32>),
    OutputSpikes(Array1<f32>)
}
