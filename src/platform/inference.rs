//! Runs an ONNX graph with Windows ML, the ONNX runtime that ships with Windows.
//!
//! Contract: a `Network` has one float input of a fixed shape and named float
//! outputs. `run` is synchronous and executes on the CPU, so results do not
//! depend on the graphics driver. What the tensors mean is the caller's business.

use std::ffi::c_void;

use windows::AI::MachineLearning::{
    LearningModel, LearningModelBinding, LearningModelDevice, LearningModelDeviceKind,
    LearningModelSession, LearningModelSessionOptions, TensorFloat,
};
use windows::Storage::Streams::{DataWriter, IBuffer, InMemoryRandomAccessStream, RandomAccessStreamReference};
use windows::core::{GUID, HRESULT, HSTRING, IUnknown, Interface};

use crate::error::{Context, Error, Result};

/// Copies bytes into a WinRT buffer.
pub(super) fn buffer_from_bytes(bytes: &[u8]) -> Result<IBuffer> {
    let writer = DataWriter::new()?;
    writer.WriteBytes(bytes)?;
    Ok(writer.DetachBuffer()?)
}

/// `ILearningModelSessionOptionsNative::SetIntraOpNumThreadsOverride(UINT32)` and
/// `ILearningModelSessionOptionsNative1::SetIntraOpThreadSpinning(boolean)` of
/// windows.ai.machinelearning.native.h. The `windows` crate does not carry them.
const OPTIONS_NATIVE: GUID = GUID::from_u128(0xc71e953f_37b4_4564_8658_d8396866db0d);
const OPTIONS_NATIVE_1: GUID = GUID::from_u128(0x5da37a26_0526_414b_91e4_2a0fa3ddba40);

/// Keeps the runtime from occupying the processor between frames.
///
/// By default it starts one worker per core and lets each of them spin while it
/// waits for work: measured with the camera loop at 30 frames per second, seven
/// workers burned half a core each for a graph that needs a few milliseconds.
/// One thread without spinning is enough. A Windows that does not know these
/// options keeps its default; that costs processor time, not correctness.
fn calm_threads(options: &LearningModelSessionOptions) {
    set_native(options, &OPTIONS_NATIVE_1, 0u8);
    set_native(options, &OPTIONS_NATIVE, 1u32);
}

/// Calls the one method of a native options interface, if this Windows has it.
fn set_native<T: Copy>(options: &LearningModelSessionOptions, interface: &GUID, value: T) {
    type Method<T> = unsafe extern "system" fn(*mut c_void, T) -> HRESULT;
    let mut raw = std::ptr::null_mut();
    // SAFETY: QueryInterface either fails or hands out one owned reference in `raw`.
    if unsafe { options.query(interface, &mut raw) }.is_err() || raw.is_null() {
        return;
    }
    // SAFETY: `raw` is that reference; `native` releases it when dropped.
    let native = unsafe { IUnknown::from_raw(raw) };
    // SAFETY: both interfaces derive from IUnknown and declare exactly one method, so it is
    // the fourth entry of the table, and `T` is the type the header gives its argument.
    unsafe {
        let table = *(raw as *const *const *const c_void);
        let method: Method<T> = std::mem::transmute_copy(&*table.add(3));
        let _ = method(native.as_raw(), value);
    }
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
        let options = LearningModelSessionOptions::new().context("create the session options")?;
        calm_threads(&options);
        let session = LearningModelSession::CreateFromModelOnDeviceWithSessionOptions(&model, &device, &options)
            .context("create the inference session")?;
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
