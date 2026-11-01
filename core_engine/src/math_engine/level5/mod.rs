//! Math engine Level 5 (Section 4.1): "Full abstracted CFR blueprint
//! strategy (card/bet abstraction + Discounted CFR solve), optionally
//! refined in real time for the current subgame."
//!
//! Confirmed with PRD author:
//!   - Card abstraction: equity-percentile bucketing (10 buckets/street
//!     default), NOT full combo tracking.
//!   - Bet abstraction: Section 3.3's discrete sizes exactly, no
//!     continuous sizing.
//!   - Streets: all 4 real streets, NOT reduced -- L5 must function as a
//!     live agent for genuine hands, not just a solver exercise. The
//!     actual tractability lever is raise-depth capping (3 raises/street
//!     default), not street-count reduction.
//!   - Real-time subgame refinement is IN SCOPE (not deferred).
//!
//! This sub-module (abstraction.rs) covers card/bet abstraction and the
//! game-tree representation only. The Discounted CFR solve loop and
//! real-time refinement are separate sub-modules, built in later messages
//! on top of what's here.

pub mod abstraction;
pub mod cfr;

pub use abstraction::{
    ActionAbstraction, BettingNode, BucketConfig, CFRConfig, InfoSetKey, StreetBuckets,
};
pub use cfr::{BestResponseConfig, CFRSolver, TrainingTrace};
