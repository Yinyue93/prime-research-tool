//! Distributed computing support for prime research
//!
//! This module provides functionality for coordinating prime research tasks
//! across multiple network nodes using Tokio for async communication.

use crate::error::{PrimeError, Result};
use crate::sieve::sieve_range;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::{timeout, Duration};

/// Work unit for distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    pub id: String,
    pub range: Range<u64>,
    pub task_type: TaskType,
    pub priority: u8,
    pub estimated_duration_ms: u64,
}

/// Types of tasks that can be distributed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    PrimeSieve,
    PrimalityTest { numbers: Vec<u64> },
    Factorization { numbers: Vec<u64> },
    GapAnalysis,
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

/// Distributed coordinator for managing work across nodes
pub struct DistributedCoordinator {
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    work_queue: Arc<Mutex<Vec<WorkUnit>>>,
    pending_work: Arc<RwLock<HashMap<String, WorkUnit>>>,
    completed_work: Arc<Mutex<Vec<WorkResult>>>,
    node_timeout: Duration,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub fn new(node_timeout_seconds: u64) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            work_queue: Arc::new(Mutex::new(Vec::new())),
            pending_work: Arc::new(RwLock::new(HashMap::new())),
            completed_work: Arc::new(Mutex::new(Vec::new())),
            node_timeout: Duration::from_secs(node_timeout_seconds),
        }
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
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }
    
    /// Get the next work unit for a node
    pub async fn get_work(&self, node_id: &str) -> Result<Option<WorkUnit>> {
        let nodes = self.nodes.read().await;
        let node = nodes.get(node_id)
            .ok_or_else(|| PrimeError::network_error("Node not registered"))?;
        
        let mut queue = self.work_queue.lock().await;
        
        // Find suitable work for this node
        let work_index = queue.iter().position(|work| {
            self.can_node_handle_task(node, &work.task_type)
        });
        
        if let Some(index) = work_index {
            let work = queue.remove(index);
            
            // Move to pending
            let mut pending = self.pending_work.write().await;
            pending.insert(work.id.clone(), work.clone());
            
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
        
        log::info!("Received result for work {} from node {}", 
                  result.work_id, result.node_id);
        
        Ok(())
    }
    
    /// Distribute prime sieving across nodes
    pub async fn distribute_sieve(&self, range: Range<u64>, chunk_size: u64) -> Result<Vec<u64>> {
        let total_range = range.end - range.start;
        let num_chunks = (total_range + chunk_size - 1) / chunk_size;
        
        // Create work units
        let mut work_units = Vec::new();
        for i in 0..num_chunks {
            let chunk_start = range.start + i * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(range.end);
            
            let work = WorkUnit {
                id: format!("sieve_{}_{}", chunk_start, chunk_end),
                range: chunk_start..chunk_end,
                task_type: TaskType::PrimeSieve,
                priority: 5,
                estimated_duration_ms: estimate_sieve_time(chunk_end - chunk_start),
            };
            
            work_units.push(work);
        }
        
        // Submit all work
        for work in work_units {
            self.submit_work(work).await?;
        }
        
        // Wait for completion and collect results
        self.wait_for_completion().await
    }
    
    /// Wait for all pending work to complete
    async fn wait_for_completion(&self) -> Result<Vec<u64>> {
        let mut all_primes = Vec::new();
        let check_interval = Duration::from_millis(100);
        let max_wait = Duration::from_secs(300); // 5 minutes max
        
        let start_time = tokio::time::Instant::now();
        
        loop {
            // Check if all work is done
            let pending_count = {
                let pending = self.pending_work.read().await;
                pending.len()
            };
            
            let queue_count = {
                let queue = self.work_queue.lock().await;
                queue.len()
            };
            
            if pending_count == 0 && queue_count == 0 {
                break;
            }
            
            // Check for timeout
            if start_time.elapsed() > max_wait {
                return Err(PrimeError::timeout(max_wait.as_secs()));
            }
            
            // Reassign timed-out work
            self.handle_timeouts().await?;
            
            tokio::time::sleep(check_interval).await;
        }
        
        // Collect all results
        let completed = self.completed_work.lock().await;
        for result in completed.iter() {
            if let TaskResult::Primes(primes) = &result.result {
                all_primes.extend(primes);
            }
        }
        
        all_primes.sort_unstable();
        Ok(all_primes)
    }
    
    /// Handle work that has timed out
    async fn handle_timeouts(&self) -> Result<()> {
        let mut timed_out_work = Vec::new();
        
        {
            let mut pending = self.pending_work.write().await;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            
            pending.retain(|_, work| {
                let elapsed = now.saturating_sub(work.estimated_duration_ms);
                if elapsed > self.node_timeout.as_millis() as u64 {
                    timed_out_work.push(work.clone());
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
            TaskType::PrimeSieve => node.cpu_cores >= 2,
            TaskType::PrimalityTest { .. } => true,
            TaskType::Factorization { .. } => node.cpu_cores >= 4,
            TaskType::GapAnalysis => node.memory_gb >= 4.0,
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
        let start_time = tokio::time::Instant::now();
        
        let result = match work.task_type {
            TaskType::PrimeSieve => {
                let primes = sieve_range(work.range);
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
            TaskType::GapAnalysis => {
                let gaps = crate::gaps::find_gaps(work.range);
                TaskResult::Gaps(gaps)
            }
        };
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        Ok(WorkResult {
            work_id: work.id,
            node_id: self.node_id.clone(),
            result,
            execution_time_ms: execution_time,
            error: None,
        })
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
            capabilities: vec![TaskType::PrimeSieve],
        };
        
        coordinator.register_node(node).await.unwrap();
        
        let work = WorkUnit {
            id: "test_work".to_string(),
            range: 1..100,
            task_type: TaskType::PrimeSieve,
            priority: 5,
            estimated_duration_ms: 1000,
        };
        
        coordinator.submit_work(work).await.unwrap();
        
        let assigned_work = coordinator.get_work("test_node").await.unwrap();
        assert!(assigned_work.is_some());
    }
    
    #[tokio::test]
    async fn test_local_executor() {
        let executor = LocalExecutor::new("test_executor".to_string());
        
        let work = WorkUnit {
            id: "sieve_test".to_string(),
            range: 1..100,
            task_type: TaskType::PrimeSieve,
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