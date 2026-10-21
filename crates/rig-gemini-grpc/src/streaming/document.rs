//! Retains the latest SDK response with SDK spellings and enums.
//! A completed stream reports its genuine terminal frame.

use crate::proto::GenerateContentResponse;
use rig_core::wire::document::Reassemble;
use serde_json::Value;

#[derive(Default)]
pub struct GrpcDocument(Value);

impl rig_core::wire::document::Serves<rig_core::operation::Completion> for GrpcDocument {}

impl Reassemble<GenerateContentResponse> for GrpcDocument {
    fn absorb(&mut self, frame: &GenerateContentResponse) {
        self.0 = super::raw::response(frame);
    }
    fn finish(self) -> Value { self.0 }
}
