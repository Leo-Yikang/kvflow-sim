//! KVFlow-Sim: request/object-level simulation utilities for LLM serving.
//!
//! The crate starts with the smallest useful layer for research:
//! request traces, model profiles, KV-cache sizing, simple compute models,
//! and object-level transfer estimates. A full serving runner can be built on
//! top of these primitives without committing to packet-level network events.

pub mod error;
pub mod model;
pub mod trace;
pub mod transfer;

pub use error::{KvFlowError, Result};
