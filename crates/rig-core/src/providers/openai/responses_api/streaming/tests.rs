//! Inherited Responses assertions at the actual driver, event, fold and history boundaries.
//! Source dispositions and unresolved behavior contrasts are recorded in
//! responses-inherited-tests-r1.md; these sources have not been executed.

mod helpers;
mod pending_contract;
mod protected_authority;
mod protected_binding;
mod protected_calls;
mod protected_evidence;
mod protected_failures;
mod protected_metadata;
mod protected_mutation;
mod protected_order;
mod protected_parts;
mod protected_protocol;
mod protected_reader;
mod protected_reasoning;
mod protected_slots;
mod protected_snapshots;
mod protected_terminal;
mod upstream;

mod terminal_call_custody;
