//! Runs an ONNX graph with Windows ML, the ONNX runtime that ships with Windows.
//!
//! Contract: a `Network` has one float input of a fixed shape and named float
//! outputs. `run` is synchronous and executes on the CPU, so results do not
//! depend on the graphics driver. What the tensors mean is the caller's business.

use windows::AI::MachineLearning::{
    LearningModel, LearningModelBinding, LearningModelDevice, LearningModelDeviceKind,
    LearningModelSession, TensorFloat,
};
use windows::Storage::Streams::{DataWriter, IBuffer, InMemoryRandomAccessStream, RandomAccessStreamReference};
use windows::core::{HSTRING, Interface};

use crate::error::{Context, Error, Result};

/// Copies bytes into a WinRT buffer.
pub(super) fn buffer_from_bytes(bytes: &[u8]) -> Result<IBuffer> {
    let writer = DataWriter::new()?;
    writer.WriteBytes(bytes)?;
    Ok(writer.DetachBuffer()?)
}

pub struct Network {
    session: LearningModelSession,
    input_name: HSTRING,
    input_shape: Vec<i64>,
    output_names: Vec<HSTRING>,
}

impl Network {
    /// Loads an ONNX model from memory. `input` is the input's name and shape;
    /// `outputs` are the names `run` fills, in order.
    pub fn load(model: &[u8], input: (&str, &[i64]), outputs: &[&str]) -> Result<Network> {
        let stream = InMemoryRandomAccessStream::new().context("create a memory stream")?;
        stream.WriteAsync(&buffer_from_bytes(model)?)?.join().context("copy the model into a stream")?;
        stream.Seek(0)?;
        let reference = RandomAccessStreamReference::CreateFromStream(&stream)?;
        let model = LearningModel::LoadFromStream(&reference).context("load the ONNX model")?;
        let device = LearningModelDevice::Create(LearningModelDeviceKind::Cpu).context("select the CPU for inference")?;
        let session =
            LearningModelSession::CreateFromModelOnDevice(&model, &device).context("create the inference session")?;
        Ok(Network {
            session,
            input_name: HSTRING::from(input.0),
            input_shape: input.1.to_vec(),
            output_names: outputs.iter().map(|name| HSTRING::from(*name)).collect(),
        })
    }

    /// Evaluates the graph. Every slice must have exactly the tensor's element count.
    pub fn run(&self, input: &[f32], outputs: &mut [&mut [f32]]) -> Result<()> {
        let expected: i64 = self.input_shape.iter().product();
        if input.len() as i64 != expected || outputs.len() != self.output_names.len() {
            return Err(Error::new("the tensors passed to the network have the wrong shape"));
        }
        let tensor = TensorFloat::CreateFromShapeArrayAndDataArray(&self.input_shape, input)?;
        let binding = LearningModelBinding::CreateFromSession(&self.session)?;
        binding.Bind(&self.input_name, &tensor)?;
        let results = self.session.Evaluate(&binding, &HSTRING::new()).context("run the network")?.Outputs()?;
        for (name, output) in self.output_names.iter().zip(outputs.iter_mut()) {
            let values = results.Lookup(name)?.cast::<TensorFloat>()?.GetAsVectorView()?;
            if values.Size()? as usize != output.len() || values.GetMany(0, output)? as usize != output.len() {
                return Err(Error::new(format!("network output '{name}' has an unexpected size")));
            }
        }
        Ok(())
    }
}
