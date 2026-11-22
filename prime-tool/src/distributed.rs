//! Distributed computing support for prime research
//!
//! This module provides a work-queue coordinator for splitting prime research
//! tasks across nodes. There is no network transport yet: in
//! [`DistributedCoordinator::distribute_sieve`] each registered node is run as
//! a local worker task (see [`LocalExecutor`]).

use crate::error::{PrimeError, Result};
use crate::sieve::sieve_range;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::time::{Duration, Instant};

/// Work unit for distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    pub id: String,
    pub task: TaskType,
    pub priority: u8,
    pub estimated_duration_ms: u64,
}

/// Types of tasks that can be distributed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    PrimeSieve { range: Range<u64> },
    PrimalityTest { numbers: Vec<u64> },
    Factorization { numbers: Vec<u64> },
    GapAnalysis { range: Range<u64> },
}

/// Result of a completed work unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    pub work_id: String,
    pub node_id: String,
    pub result: TaskResult,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// Task-specific results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResult {
    Primes(Vec<u64>),
    PrimalityResults(Vec<bool>),
    Factors(HashMap<u64, Vec<u64>>),
    Gaps(Vec<crate::gaps::GapInfo>),
}

/// Node information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub load_factor: f64,
    pub last_heartbeat: u64,
    pub capabilities: Vec<TaskType>,
}

/// Work that has been handed to a node, with the time it was assigned
#[derive(Debug, Clone)]
struct PendingWork {
    work: WorkUnit,
    assigned_at: Instant,
}

/// Distributed coordinator for managing work across nodes.
///
/// Cloning is cheap and yields a handle to the same shared state.
#[derive(Clone)]
pub struct DistributedCoordinator {
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    work_queue: Arc<Mutex<Vec<WorkUnit>>>,
    pending_work: Arc<RwLock<HashMap<String, PendingWork>>>,
    completed_work: Arc<Mutex<Vec<WorkResult>>>,
    /// Signalled whenever a result is submitted, so waiters wake immediately
    result_submitted: Arc<Notify>,
    node_timeout: Duration,
    completion_timeout: Duration,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator.
    ///
    /// Work assigned to a node is requeued if it is not completed within
    /// `node_timeout_seconds`.
    pub fn new(node_timeout_seconds: u64) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            work_queue: Arc::new(Mutex::new(Vec::new())),
            pending_work: Arc::new(RwLock::new(HashMap::new())),
            completed_work: Arc::new(Mutex::new(Vec::new())),
            result_submitted: Arc::new(Notify::new()),
            node_timeout: Duration::from_secs(node_timeout_seconds),
            completion_timeout: Duration::from_secs(300),
        }
    }

    /// Set the maximum time [`distribute_sieve`](Self::distribute_sieve)
    /// waits for all work to finish (default 300 s).
    pub fn with_completion_timeout(mut self, timeout: Duration) -> Self {
        self.completion_timeout = timeout;
        self
    }

    /// Register a new node
    pub async fn register_node(&self, node: NodeInfo) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        log::info!("Registering node: {} at {}", node.id, node.address);
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Remove a node
    pub async fn unregister_node(&self, node_id: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if nodes.remove(node_id).is_some() {
            log::info!("Unregistered node: {}", node_id);
        }
        Ok(())
    }

    /// Add work to the queue
    pub async fn submit_work(&self, work: WorkUnit) -> Result<()> {
        let mut queue = self.work_queue.lock().await;
        queue.push(work);
        queue.sort_by_key(|a| std::cmp::Reverse(a.priority));
        Ok(())
    }

    /// Get the next work unit for a node
    pub async fn get_work(&self, node_id: &str) -> Result<Option<WorkUnit>> {
        let nodes = self.nodes.read().await;
        let node = nodes
            .get(node_id)
            .ok_or_else(|| PrimeError::network_error("Node not registered"))?;

        let mut queue = self.work_queue.lock().await;

        // Find suitable work for this node
        let work_index = queue
            .iter()
            .position(|work| self.can_node_handle_task(node, &work.task));

        if let Some(index) = work_index {
            let work = queue.remove(index);

            // Move to pending, recording when it was assigned
            let mut pending = self.pending_work.write().await;
            pending.insert(
                work.id.clone(),
                PendingWork {
                    work: work.clone(),
                    assigned_at: Instant::now(),
                },
            );

            log::info!("Assigned work {} to node {}", work.id, node_id);
            Ok(Some(work))
        } else {
            Ok(None)
        }
    }

    /// Submit completed work
    pub async fn submit_result(&self, result: WorkResult) -> Result<()> {
        // Remove from pending
        let mut pending = self.pending_work.write().await;
        pending.remove(&result.work_id);

        // Add to completed
        let mut completed = self.completed_work.lock().await;
        completed.push(result.clone());
        drop(completed);
        drop(pending);
        self.result_submitted.notify_waiters();

        log::info!(
            "Received result for work {} from node {}",
            result.work_id,
            result.node_id
        );

        Ok(())
    }

    /// Distribute prime sieving across the registered nodes.
    ///
    /// The range is split into `chunk_size` work units. Each registered node is
    /// run as a local worker task that pulls work from the queue, executes it
    /// with [`LocalExecutor`] on a blocking thread and submits the result.
    /// Returns the sorted primes, or an error if no nodes are registered or the
    /// work does not finish within the completion timeout.
    pub async fn distribute_sieve(&self, range: Range<u64>, chunk_size: u64) -> Result<Vec<u64>> {
        let node_ids: Vec<String> = self.nodes.read().await.keys().cloned().collect();
        if node_ids.is_empty() {
            return Err(PrimeError::network_error(
                "No nodes registered; cannot distribute work",
            ));
        }
        if range.start >= range.end {
            return Ok(Vec::new());
        }
        let chunk_size = chunk_size.max(1);
        let total_range = range.end - range.start;
        let num_chunks = total_range.div_ceil(chunk_size);

        // Create work units
        let mut work_units = Vec::new();
        for i in 0..num_chunks {
            let chunk_start = range.start + i * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(range.end);

            let work = WorkUnit {
                id: format!("sieve_{}_{}", chunk_start, chunk_end),
                task: TaskType::PrimeSieve {
                    range: chunk_start..chunk_end,
                },
                priority: 5,
                estimated_duration_ms: estimate_sieve_time(chunk_end - chunk_start),
            };

            work_units.push(work);
        }

        // Submit all work
        let work_ids: HashSet<String> = work_units.iter().map(|w| w.id.clone()).collect();
        for work in work_units {
            self.submit_work(work).await?;
        }

        // Run one local worker per registered node
        let workers: Vec<_> = node_ids
            .into_iter()
            .map(|node_id| tokio::spawn(self.clone().run_local_worker(node_id)))
            .collect();

        let result = self.wait_for_completion(&work_ids).await;
        for worker in workers {
            worker.abort();
        }
        result
    }

    /// Pull and execute work as `node_id` until the queue has no work it can take.
    async fn run_local_worker(self, node_id: String) -> Result<()> {
        let executor = Arc::new(LocalExecutor::new(node_id.clone()));
        while let Some(work) = self.get_work(&node_id).await? {
            let executor = Arc::clone(&executor);
            let result = tokio::task::spawn_blocking(move || executor.execute(work))
                .await
                .map_err(|e| PrimeError::network_error(format!("worker task failed: {}", e)))?;
            self.submit_result(result).await?;
        }
        Ok(())
    }

    /// Wait until every unit in `work_ids` has a result, then return the
    /// merged, sorted primes. Results are removed from the completed list.
    async fn wait_for_completion(&self, work_ids: &HashSet<String>) -> Result<Vec<u64>> {
        let mut all_primes = Vec::new();
        // Waiters are woken by submit_result; the interval only bounds how
        // often timed-out work is checked for.
        let check_interval = Duration::from_millis(100);
        let max_wait = self.completion_timeout;

        let start_time = Instant::now();

        loop {
            // Created before checking, so a result submitted in between still
            // wakes us (notify_waiters reaches futures that already exist).
            let notified = self.result_submitted.notified();

            // Check if all work is done
            let done = {
                let completed = self.completed_work.lock().await;
                let finished: HashSet<&str> = completed
                    .iter()
                    .filter(|r| work_ids.contains(&r.work_id))
                    .map(|r| r.work_id.as_str())
                    .collect();
                finished.len() == work_ids.len()
            };

            if done {
                break;
            }

            // Check for timeout
            if start_time.elapsed() > max_wait {
                return Err(PrimeError::timeout(max_wait.as_secs()));
            }

            // Reassign timed-out work
            self.handle_timeouts().await?;

            let _ = tokio::time::timeout(check_interval, notified).await;
        }

        // Drop leftover copies of this call's units: a unit that timed out and
        // was requeued may still be queued or pending after the original finished.
        self.work_queue
            .lock()
            .await
            .retain(|w| !work_ids.contains(&w.id));
        self.pending_work
            .write()
            .await
            .retain(|id, _| !work_ids.contains(id));

        // Collect this call's results (once per work unit: a requeued unit
        // can complete twice) and leave other results in place.
        let mut completed = self.completed_work.lock().await;
        let mut seen = HashSet::new();
        completed.retain(|result| {
            if !work_ids.contains(&result.work_id) {
                return true;
            }
            if seen.insert(result.work_id.clone()) {
                if let TaskResult::Primes(primes) = &result.result {
                    all_primes.extend(primes);
                }
            }
            false
        });

        all_primes.sort_unstable();
        Ok(all_primes)
    }

    /// Handle work that has timed out
    async fn handle_timeouts(&self) -> Result<()> {
        let mut timed_out_work = Vec::new();

        {
            let mut pending = self.pending_work.write().await;
            pending.retain(|_, pending_work| {
                // Time since this unit was handed to a node
                if pending_work.assigned_at.elapsed() > self.node_timeout {
                    timed_out_work.push(pending_work.work.clone());
                    false
                } else {
                    true
                }
            });
        }

        // Requeue timed-out work
        for work in timed_out_work {
            log::warn!("Work {} timed out, requeuing", work.id);
            self.submit_work(work).await?;
        }

        Ok(())
    }

    /// Check if a node can handle a specific task type
    fn can_node_handle_task(&self, node: &NodeInfo, task_type: &TaskType) -> bool {
        // Simple capability matching
        match task_type {
            TaskType::PrimeSieve { .. } => node.cpu_cores >= 2,
            TaskType::PrimalityTest { .. } => true,
            TaskType::Factorization { .. } => node.cpu_cores >= 4,
            TaskType::GapAnalysis { .. } => node.memory_gb >= 4.0,
        }
    }

    /// Get cluster statistics
    pub async fn get_cluster_stats(&self) -> ClusterStats {
        let nodes = self.nodes.read().await;
        let pending = self.pending_work.read().await;
        let queue = self.work_queue.lock().await;
        let completed = self.completed_work.lock().await;

        ClusterStats {
            total_nodes: nodes.len(),
            active_nodes: nodes.values().filter(|n| n.load_factor < 0.9).count(),
            pending_work: pending.len(),
            queued_work: queue.len(),
            completed_work: completed.len(),
            total_cpu_cores: nodes.values().map(|n| n.cpu_cores).sum(),
            total_memory_gb: nodes.values().map(|n| n.memory_gb).sum(),
        }
    }
}

/// Cluster statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub pending_work: usize,
    pub queued_work: usize,
    pub completed_work: usize,
    pub total_cpu_cores: usize,
    pub total_memory_gb: f64,
}

/// Estimate sieving time based on range size
fn estimate_sieve_time(range_size: u64) -> u64 {
    // Rough estimate: 1 microsecond per number
    range_size
}

/// Create work units for distributed sieve computation
pub fn create_sieve_work_units(range: Range<u64>, num_workers: usize) -> Vec<WorkUnit> {
    if num_workers == 0 {
        return vec![];
    }

    let total_range = range.end - range.start;
    let chunk_size = (total_range / num_workers as u64).max(1);
    let mut work_units = Vec::new();

    for i in 0..num_workers {
        let start = range.start + i as u64 * chunk_size;
        let end = if i == num_workers - 1 {
            range.end
        } else {
            (start + chunk_size).min(range.end)
        };

        if start < end {
            let work_range = start..end;
            let range_size = end - start;

            work_units.push(WorkUnit {
                id: format!("sieve_{}_{}", start, end),
                task: TaskType::PrimeSieve { range: work_range },
                priority: 1,
                estimated_duration_ms: estimate_sieve_time(range_size),
            });
        }
    }

    work_units
}

/// Local work executor for processing tasks
pub struct LocalExecutor {
    node_id: String,
}

impl LocalExecutor {
    pub fn new(node_id: String) -> Self {
        Self { node_id }
    }

    /// Execute a work unit locally
    pub async fn execute_work(&self, work: WorkUnit) -> Result<WorkResult> {
        Ok(self.execute(work))
    }

    /// Execute a work unit on the current thread (CPU-bound; call from a
    /// blocking context such as `tokio::task::spawn_blocking`).
    pub fn execute(&self, work: WorkUnit) -> WorkResult {
        let start_time = std::time::Instant::now();

        let result = match work.task {
            TaskType::PrimeSieve { range } => {
                let primes = sieve_range(range);
                TaskResult::Primes(primes)
            }
            TaskType::PrimalityTest { numbers } => {
                let results = numbers.iter().map(|&n| crate::is_prime(n)).collect();
                TaskResult::PrimalityResults(results)
            }
            TaskType::Factorization { numbers } => {
                let mut factors = HashMap::new();
                for &n in &numbers {
                    factors.insert(n, crate::factor(n));
                }
                TaskResult::Factors(factors)
            }
            TaskType::GapAnalysis { range } => {
                let gaps = crate::gaps::find_gaps(range);
                TaskResult::Gaps(gaps)
            }
        };

        let execution_time = start_time.elapsed().as_millis() as u64;

        WorkResult {
            work_id: work.id,
            node_id: self.node_id.clone(),
            result,
            execution_time_ms: execution_time,
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_basic() {
        let coordinator = DistributedCoordinator::new(30);

        let node = NodeInfo {
            id: "test_node".to_string(),
            address: "localhost:8080".to_string(),
            cpu_cores: 4,
            memory_gb: 8.0,
            load_factor: 0.5,
            last_heartbeat: 0,
            capabilities: vec![
                TaskType::PrimeSieve { range: 0..0 },
                TaskType::GapAnalysis { range: 0..0 },
            ],
        };

        coordinator.register_node(node).await.unwrap();

        let work = WorkUnit {
            id: "test_work".to_string(),
            task: TaskType::PrimeSieve { range: 1..100 },
            priority: 5,
            estimated_duration_ms: 1000,
        };

        coordinator.submit_work(work).await.unwrap();

        let assigned_work = coordinator.get_work("test_node").await.unwrap();
        assert!(assigned_work.is_some());
    }

    fn node(id: &str) -> NodeInfo {
        NodeInfo {
            id: id.to_string(),
            address: "127.0.0.1:0".to_string(),
            cpu_cores: 4,
            memory_gb: 8.0,
            load_factor: 0.0,
            last_heartbeat: 0,
            capabilities: vec![],
        }
    }

    #[tokio::test]
    async fn test_distribute_sieve_completes_and_matches_local() {
        let coordinator =
            DistributedCoordinator::new(30).with_completion_timeout(Duration::from_secs(20));
        for i in 0..4 {
            coordinator
                .register_node(node(&format!("node_{}", i)))
                .await
                .unwrap();
        }
        let primes = coordinator
            .distribute_sieve(1..100_000, 7_000)
            .await
            .unwrap();
        assert_eq!(primes, sieve_range(1..100_000));

        // A second call on the same coordinator returns only its own results
        let primes = coordinator.distribute_sieve(1..1_000, 100).await.unwrap();
        assert_eq!(primes, sieve_range(1..1_000));

        let stats = coordinator.get_cluster_stats().await;
        assert_eq!(
            (stats.pending_work, stats.queued_work, stats.completed_work),
            (0, 0, 0)
        );
    }

    #[tokio::test]
    async fn test_distribute_sieve_without_nodes_errors() {
        let coordinator = DistributedCoordinator::new(30);
        assert!(coordinator.distribute_sieve(1..1_000, 100).await.is_err());
    }

    #[tokio::test]
    async fn test_timed_out_work_is_requeued() {
        let coordinator = DistributedCoordinator::new(0);
        coordinator.register_node(node("slow")).await.unwrap();
        coordinator
            .submit_work(WorkUnit {
                id: "w".to_string(),
                task: TaskType::PrimeSieve { range: 1..10 },
                priority: 1,
                estimated_duration_ms: 10_000_000,
            })
            .await
            .unwrap();
        assert!(coordinator.get_work("slow").await.unwrap().is_some());
        tokio::time::sleep(Duration::from_millis(5)).await;
        coordinator.handle_timeouts().await.unwrap();
        let stats = coordinator.get_cluster_stats().await;
        assert_eq!((stats.pending_work, stats.queued_work), (0, 1));
    }

    #[tokio::test]
    async fn test_recent_work_is_not_requeued() {
        // Regression: the old check compared "now - estimated_duration_ms"
        // (an epoch timestamp) to the timeout, so all work looked timed out.
        let coordinator = DistributedCoordinator::new(30);
        coordinator.register_node(node("n")).await.unwrap();
        coordinator
            .submit_work(WorkUnit {
                id: "w".to_string(),
                task: TaskType::PrimeSieve { range: 1..10 },
                priority: 1,
                estimated_duration_ms: 10,
            })
            .await
            .unwrap();
        coordinator.get_work("n").await.unwrap();
        coordinator.handle_timeouts().await.unwrap();
        let stats = coordinator.get_cluster_stats().await;
        assert_eq!((stats.pending_work, stats.queued_work), (1, 0));
    }

    #[tokio::test]
    async fn test_local_executor() {
        let executor = LocalExecutor::new("test_executor".to_string());

        let work = WorkUnit {
            id: "sieve_test".to_string(),
            task: TaskType::PrimeSieve { range: 1..100 },
            priority: 5,
            estimated_duration_ms: 1000,
        };

        let result = executor.execute_work(work).await.unwrap();

        match result.result {
            TaskResult::Primes(primes) => {
                assert!(primes.contains(&97));
            }
            _ => panic!("Expected primes result"),
        }
    }
}
