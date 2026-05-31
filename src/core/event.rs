use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A generic discrete-event for time-driven simulation.
///
/// Events are ordered by `time_ns` ascending; a monotonic `seq` breaks ties
/// so that two events with the same time are not considered equal (which
/// would violate `BinaryHeap` invariants).
#[derive(Debug)]
pub struct Event<T> {
    pub time_ns: u64,
    pub seq: u64,
    pub kind: T,
}

impl<T> PartialEq for Event<T> {
    fn eq(&self, other: &Self) -> bool {
        self.time_ns == other.time_ns && self.seq == other.seq
    }
}

impl<T> Eq for Event<T> {}

impl<T> PartialOrd for Event<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Event<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap pops the *smallest* time first.
        other
            .time_ns
            .cmp(&self.time_ns)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Priority queue for discrete-event simulation.
#[derive(Debug)]
pub struct EventQueue<T> {
    queue: BinaryHeap<Event<T>>,
    next_seq: u64,
}

impl<T> Default for EventQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> EventQueue<T> {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            next_seq: 0,
        }
    }

    /// Schedule an event at `time_ns`.  If `time_ns` is in the past the event
    /// is still enqueued and will be processed immediately on the next step.
    pub fn schedule(&mut self, time_ns: u64, kind: T) {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        self.queue.push(Event { time_ns, seq, kind });
    }

    /// Remove and return the earliest event.
    pub fn pop(&mut self) -> Option<Event<T>> {
        self.queue.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Peek at the earliest event without removing it.
    pub fn peek(&self) -> Option<&Event<T>> {
        self.queue.peek()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_processed_in_time_order() {
        let mut q = EventQueue::new();
        q.schedule(100, "c");
        q.schedule(10, "a");
        q.schedule(50, "b");

        assert_eq!(q.pop().unwrap().kind, "a");
        assert_eq!(q.pop().unwrap().kind, "b");
        assert_eq!(q.pop().unwrap().kind, "c");
        assert!(q.is_empty());
    }

    #[test]
    fn same_time_processed_fifo() {
        let mut q = EventQueue::new();
        q.schedule(10, "first");
        q.schedule(10, "second");

        assert_eq!(q.pop().unwrap().kind, "first");
        assert_eq!(q.pop().unwrap().kind, "second");
    }
}
