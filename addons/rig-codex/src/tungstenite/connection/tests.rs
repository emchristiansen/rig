use super::*;
use bytes::Bytes;

/// Every frame maps to the tungstenite message of its kind and back, payload
/// intact.
#[test]
fn every_frame_maps_to_its_message_and_back() {
    for frame in [
        Frame::Text("{ \"type\" : \"x\" }".to_owned()),
        Frame::Binary(Bytes::from_static(&[0xff, 0x00, 0x7b])),
        Frame::Ping(Bytes::from_static(b"ping")),
        Frame::Pong(Bytes::from_static(b"pong")),
        Frame::Close(None),
        Frame::Close(Some(CloseFrame {
            code: 1011,
            reason: "server restarting".to_owned(),
        })),
    ] {
        assert_eq!(
            from_message(into_message(frame.clone())),
            Some(frame.clone()),
            "{frame:?}"
        );
    }
}

/// A raw frame carries no protocol payload and is not handed to a session.
#[test]
fn a_raw_frame_is_not_a_session_frame() {
    use tungstenite::protocol::frame::coding::{Data, OpCode};
    use tungstenite::protocol::frame::{Frame as RawFrame, FrameHeader};

    let raw = RawFrame::from_payload(
        FrameHeader {
            is_final: false,
            opcode: OpCode::Data(Data::Text),
            ..FrameHeader::default()
        },
        Bytes::from_static(b"partial"),
    );
    assert_eq!(from_message(Message::Frame(raw)), None);
}
