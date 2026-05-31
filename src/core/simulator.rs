use super::EventQueue;

/// A minimal discrete-event simulator shell.
///
/// Users drive the loop by repeatedly calling `step` and handling the returned
/// event.  This keeps the simulator agnostic to the event kind.
#[derive(Debug)]
pub struct Simulator<T> {
    now_ns: u64,
    queue: EventQueue<T>,
}

impl<T> Default for Simulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Simulator<T> {
    pub fn new() -> Self {
        Self {
            now_ns: 0,
            queue: EventQueue::new(),
        }
    }

    pub fn now_ns(&self) -> u64 {
        self.now_ns
    }

    pub fn schedule(&mut self, time_ns: u64, kind: T) {
        self.queue.schedule(time_ns, kind);
    }

    /// Advance the clock to the next event time and return it.
    pub fn step(&mut self) -> Option<(u64, T)> {
        let ev = self.queue.pop()?;
        self.now_ns = self.now_ns.max(ev.time_ns);
        Some((self.now_ns, ev.kind))
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_advances_clock() {
        let mut sim = Simulator::new();
        sim.schedule(100, "e1");
        sim.schedule(50, "e2");

        assert_eq!(sim.now_ns(), 0);
        assert_eq!(sim.step(), Some((50, "e2")));
        assert_eq!(sim.now_ns(), 50);
        assert_eq!(sim.step(), Some((100, "e1")));
        assert_eq!(sim.now_ns(), 100);
        assert!(sim.is_empty());
    }
}
