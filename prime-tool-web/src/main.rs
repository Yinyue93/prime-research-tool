//! Prime Research Tool Web Service
//! 
//! A REST API service built with Axum that exposes prime research functionality
//! over HTTP endpoints.

use axum::{
    extract::{Query, Path},
    http::StatusCode,
    response::{Json, Result as AxumResult},
    routing::{get, post},
    Router,
};
use prime_tool::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing::{info, warn};
use tracing_subscriber;
use chrono;
use rand::Rng;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting Prime Research Tool Web Service");
    
    // Build the application router
    let app = create_app();
    
    // Bind to address
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let listener = TcpListener::bind(addr).await?;
    
    info!("Server listening on http://{}", addr);
    info!("API Documentation available at http://{}/docs", addr);
    
    // Start the server
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Create the Axum application with all routes
fn create_app() -> Router {
    Router::new()
        // Health check
        .route("/health", get(health_check))
        
        // API documentation
        .route("/docs", get(api_docs))
        
        // Primality testing
        .route("/prime/:number", get(check_prime))
        .route("/prime", post(check_prime_batch))
        
        // Factorization
        .route("/factor/:number", get(factor_number))
        .route("/factor", post(factor_batch))
        
        // Sieving
        .route("/sieve", get(sieve_range_handler))
        .route("/sieve/parallel", get(sieve_parallel_handler))
        
        // Gap analysis
        .route("/gaps", get(gap_analysis_handler))
        .route("/gaps/twins", get(twin_primes_handler))
        .route("/gaps/cousins", get(cousin_primes_handler))
        .route("/gaps/sexy", get(sexy_primes_handler))
        
        // Statistics and analysis
        .route("/stats/range", get(range_statistics))
        .route("/benchmark", post(run_benchmark))
        
        // Add middleware
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        )
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
struct RangeQuery {
    start: u64,
    end: u64,
    #[serde(default)]
    chunk_size: Option<usize>,
}

#[derive(Deserialize)]
struct BatchRequest {
    numbers: Vec<u64>,
}

#[derive(Deserialize)]
struct BenchmarkRequest {
    max_n: u64,
    iterations: usize,
    benchmark_type: String, // "primality", "factorization", "sieve", "gaps"
}

#[derive(Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
    execution_time_ms: u64,
}

#[derive(Serialize, Deserialize)]
struct PrimalityResponse {
    number: u64,
    is_prime: bool,
    algorithm_used: String,
}

#[derive(Serialize, Deserialize)]
struct FactorizationResponse {
    number: u64,
    factors: Vec<u64>,
    is_prime: bool,
}

#[derive(Serialize, Deserialize)]
struct SieveResponse {
    range_start: u64,
    range_end: u64,
    primes: Vec<u64>,
    count: usize,
    method: String,
}

#[derive(Serialize)]
struct GapAnalysisResponse {
    range_start: u64,
    range_end: u64,
    total_gaps: usize,
    max_gap: u64,
    average_gap: f64,
    gap_distribution: HashMap<u64, usize>,
}

#[derive(Serialize)]
struct TwinPrimesResponse {
    range_start: u64,
    range_end: u64,
    twin_primes: Vec<(u64, u64)>,
    count: usize,
}

#[derive(Serialize)]
struct RangeStatsResponse {
    range_start: u64,
    range_end: u64,
    prime_count: usize,
    density: f64,
    largest_prime: u64,
    largest_gap: u64,
    twin_prime_count: usize,
}

#[derive(Serialize)]
struct BenchmarkResponse {
    benchmark_type: String,
    max_n: u64,
    iterations: usize,
    total_time_ms: u64,
    average_time_ns: f64,
    operations_per_second: f64,
}

// ============================================================================
// Handler Functions
// ============================================================================

/// Health check endpoint
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "Prime Research Tool API",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

/// API documentation endpoint
async fn api_docs() -> &'static str {
    r#"
<!DOCTYPE html>
<html>
<head>
    <title>Prime Research Tool API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .method { font-weight: bold; color: #007acc; }
        .path { font-family: monospace; background: #f5f5f5; padding: 2px 5px; }
        pre { background: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Prime Research Tool API Documentation</h1>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/health</span></h3>
        <p>Health check endpoint</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/prime/{number}</span></h3>
        <p>Check if a number is prime</p>
        <pre>Example: GET /prime/999983</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">POST</span> <span class="path">/prime</span></h3>
        <p>Check multiple numbers for primality</p>
        <pre>Body: {"numbers": [999983, 1000003, 1000033]}</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/factor/{number}</span></h3>
        <p>Factor a number</p>
        <pre>Example: GET /factor/1234567</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/sieve?start=1&end=1000</span></h3>
        <p>Find all primes in a range using sequential sieve</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/sieve/parallel?start=1&end=1000000&chunk_size=10000</span></h3>
        <p>Find all primes in a range using parallel sieve</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/gaps?start=1&end=1000</span></h3>
        <p>Analyze prime gaps in a range</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/gaps/twins?start=1&end=1000</span></h3>
        <p>Find twin primes in a range</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> <span class="path">/stats/range?start=1&end=1000</span></h3>
        <p>Get comprehensive statistics for a range</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">POST</span> <span class="path">/benchmark</span></h3>
        <p>Run performance benchmarks</p>
        <pre>Body: {"max_n": 100000, "iterations": 1000, "benchmark_type": "primality"}</pre>
    </div>
</body>
</html>
    "#
}

/// Check if a single number is prime
async fn check_prime(Path(number): Path<u64>) -> AxumResult<Json<ApiResponse<PrimalityResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    match tokio::task::spawn_blocking(move || {
        let is_prime_result = is_prime(number);
        let algorithm = if number <= u64::MAX {
            "Miller-Rabin"
        } else {
            "Baillie-PSW"
        };
        
        PrimalityResponse {
            number,
            is_prime: is_prime_result,
            algorithm_used: algorithm.to_string(),
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in primality test: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Check multiple numbers for primality
async fn check_prime_batch(
    Json(request): Json<BatchRequest>,
) -> AxumResult<Json<ApiResponse<Vec<PrimalityResponse>>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if request.numbers.len() > 1000 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Too many numbers (max 1000)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    match tokio::task::spawn_blocking(move || {
        request.numbers.into_iter().map(|number| {
            let is_prime_result = is_prime(number);
            let algorithm = if number <= u64::MAX {
                "Miller-Rabin"
            } else {
                "Baillie-PSW"
            };
            
            PrimalityResponse {
                number,
                is_prime: is_prime_result,
                algorithm_used: algorithm.to_string(),
            }
        }).collect::<Vec<_>>()
    }).await {
        Ok(results) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(results),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in batch primality test: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Factor a single number
async fn factor_number(Path(number): Path<u64>) -> AxumResult<Json<ApiResponse<FactorizationResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    match tokio::task::spawn_blocking(move || {
        let factors = factor(number);
        let is_prime_result = factors.len() == 1 && factors[0] == number;
        
        FactorizationResponse {
            number,
            factors,
            is_prime: is_prime_result,
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in factorization: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Factor multiple numbers
async fn factor_batch(
    Json(request): Json<BatchRequest>,
) -> AxumResult<Json<ApiResponse<Vec<FactorizationResponse>>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if request.numbers.len() > 100 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Too many numbers for factorization (max 100)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    match tokio::task::spawn_blocking(move || {
        request.numbers.into_iter().map(|number| {
            let factors = factor(number);
            let is_prime_result = factors.len() == 1 && factors[0] == number;
            
            FactorizationResponse {
                number,
                factors,
                is_prime: is_prime_result,
            }
        }).collect::<Vec<_>>()
    }).await {
        Ok(results) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(results),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in batch factorization: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Sieve primes in a range (sequential)
async fn sieve_range_handler(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<SieveResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if params.end - params.start > 10_000_000 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Range too large (max 10M)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    match tokio::task::spawn_blocking(move || {
        let primes = sieve_range(params.start..params.end);
        SieveResponse {
            range_start: params.start,
            range_end: params.end,
            count: primes.len(),
            primes,
            method: "Sequential".to_string(),
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in sieve: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Sieve primes in a range (parallel)
async fn sieve_parallel_handler(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<SieveResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if params.end - params.start > 10_000_000 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Range too large (max 10M)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    let chunk_size = params.chunk_size.unwrap_or(10_000);
    
    match tokio::task::spawn_blocking(move || {
        let primes = sieve_parallel(params.start..params.end, chunk_size);
        SieveResponse {
            range_start: params.start,
            range_end: params.end,
            count: primes.len(),
            primes,
            method: "Parallel".to_string(),
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in parallel sieve: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Analyze prime gaps in a range
async fn gap_analysis_handler(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<GapAnalysisResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if params.end - params.start > 1_000_000 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Range too large for gap analysis (max 1M)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    match tokio::task::spawn_blocking(move || {
        let stats = gaps::analyze_gaps(params.start..params.end);
        
        GapAnalysisResponse {
            range_start: params.start,
            range_end: params.end,
            total_gaps: stats.total_gaps,
            max_gap: stats.max_gap as u64,
            average_gap: stats.average_gap,
            gap_distribution: stats.gap_histogram.into_iter().map(|(k, v)| (k as u64, v)).collect(),
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error in gap analysis: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Find twin primes in a range
async fn twin_primes_handler(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<TwinPrimesResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    match tokio::task::spawn_blocking(move || {
        let twins = gaps::find_twin_primes(params.start..params.end);
        let twin_pairs: Vec<(u64, u64)> = twins.into_iter()
            .map(|tp| (tp.smaller, tp.larger))
            .collect();
        
        TwinPrimesResponse {
            range_start: params.start,
            range_end: params.end,
            count: twin_pairs.len(),
            twin_primes: twin_pairs,
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error finding twin primes: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Find cousin primes in a range
async fn cousin_primes_handler(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<TwinPrimesResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    match tokio::task::spawn_blocking(move || {
        let cousins = gaps::find_cousin_primes(params.start..params.end);
        let cousin_pairs: Vec<(u64, u64)> = cousins.into_iter()
            .map(|tp| (tp.smaller, tp.larger))
            .collect();
        
        TwinPrimesResponse {
            range_start: params.start,
            range_end: params.end,
            count: cousin_pairs.len(),
            twin_primes: cousin_pairs, // Reusing the same structure
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error finding cousin primes: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Find sexy primes in a range
async fn sexy_primes_handler(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<TwinPrimesResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    match tokio::task::spawn_blocking(move || {
        let sexy = gaps::find_sexy_primes(params.start..params.end);
        let sexy_pairs: Vec<(u64, u64)> = sexy.into_iter()
            .map(|tp| (tp.smaller, tp.larger))
            .collect();
        
        TwinPrimesResponse {
            range_start: params.start,
            range_end: params.end,
            count: sexy_pairs.len(),
            twin_primes: sexy_pairs, // Reusing the same structure
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error finding sexy primes: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get comprehensive range statistics
async fn range_statistics(
    Query(params): Query<RangeQuery>,
) -> AxumResult<Json<ApiResponse<RangeStatsResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if params.end - params.start > 1_000_000 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Range too large for statistics (max 1M)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    match tokio::task::spawn_blocking(move || {
        let primes = sieve_range(params.start..params.end);
        let gaps = gaps::find_gaps(params.start..params.end);
        let twins = gaps::find_twin_primes(params.start..params.end);
        
        let largest_prime = primes.last().copied().unwrap_or(0);
        let largest_gap = gaps.iter().map(|g| g.gap_size).max().unwrap_or(0);
        let range_size = params.end - params.start;
        let density = primes.len() as f64 / range_size as f64;
        
        RangeStatsResponse {
            range_start: params.start,
            range_end: params.end,
            prime_count: primes.len(),
            density,
            largest_prime,
            largest_gap: largest_gap as u64,
            twin_prime_count: twins.len(),
        }
    }).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Err(e) => {
            warn!("Error computing range statistics: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Run performance benchmarks
async fn run_benchmark(
    Json(request): Json<BenchmarkRequest>,
) -> AxumResult<Json<ApiResponse<BenchmarkResponse>>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    if request.iterations > 10_000 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Too many iterations (max 10,000)".to_string()),
            execution_time_ms: 0,
        }));
    }
    
    match tokio::task::spawn_blocking(move || {
        let bench_start = std::time::Instant::now();
        
        match request.benchmark_type.as_str() {
            "primality" => {
                // Simple primality benchmark
                let mut rng = rand::thread_rng();
                for _ in 0..request.iterations {
                    let n = rng.gen_range(2..=request.max_n);
                    is_prime(n);
                }
            }
            "factorization" => {
                // Simple factorization benchmark
                let mut rng = rand::thread_rng();
                for _ in 0..request.iterations {
                    let n = rng.gen_range(4..=request.max_n);
                    factor(n);
                }
            }
            "sieve" => {
                // Sieve benchmark
                for _ in 0..request.iterations {
                    let end = std::cmp::min(request.max_n, 10_000);
                    sieve_range(1..end);
                }
            }
            _ => {
                return Err("Invalid benchmark type");
            }
        }
        
        let total_time = bench_start.elapsed();
        let average_time_ns = total_time.as_nanos() as f64 / request.iterations as f64;
        let ops_per_second = 1_000_000_000.0 / average_time_ns;
        
        Ok(BenchmarkResponse {
            benchmark_type: request.benchmark_type,
            max_n: request.max_n,
            iterations: request.iterations,
            total_time_ms: total_time.as_millis() as u64,
            average_time_ns,
            operations_per_second: ops_per_second,
        })
    }).await {
        Ok(Ok(result)) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                error: None,
                execution_time_ms: execution_time,
            }))
        }
        Ok(Err(e)) => {
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(e.to_string()),
                execution_time_ms: 0,
            }))
        }
        Err(e) => {
            warn!("Error in benchmark: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    
    #[tokio::test]
    async fn test_health_check() {
        let app = create_app();
        let server = TestServer::new(app).unwrap();
        
        let response = server.get("/health").await;
        response.assert_status_ok();
        
        let json: serde_json::Value = response.json();
        assert_eq!(json["status"], "healthy");
    }
    
    #[tokio::test]
    async fn test_prime_check() {
        let app = create_app();
        let server = TestServer::new(app).unwrap();
        
        let response = server.get("/prime/17").await;
        response.assert_status_ok();
        
        let json: ApiResponse<PrimalityResponse> = response.json();
        assert!(json.success);
        assert!(json.data.unwrap().is_prime);
    }
    
    #[tokio::test]
    async fn test_factorization() {
        let app = create_app();
        let server = TestServer::new(app).unwrap();
        
        let response = server.get("/factor/15").await;
        response.assert_status_ok();
        
        let json: ApiResponse<FactorizationResponse> = response.json();
        assert!(json.success);
        let data = json.data.unwrap();
        assert_eq!(data.number, 15);
        assert!(!data.is_prime);
        assert!(data.factors.contains(&3) && data.factors.contains(&5));
    }
    
    #[tokio::test]
    async fn test_sieve() {
        let app = create_app();
        let server = TestServer::new(app).unwrap();
        
        let response = server.get("/sieve?start=1&end=20").await;
        response.assert_status_ok();
        
        let json: ApiResponse<SieveResponse> = response.json();
        assert!(json.success);
        let data = json.data.unwrap();
        assert!(data.primes.contains(&2));
        assert!(data.primes.contains(&3));
        assert!(data.primes.contains(&5));
        assert!(data.primes.contains(&7));
        assert!(data.primes.contains(&11));
        assert!(data.primes.contains(&13));
        assert!(data.primes.contains(&17));
        assert!(data.primes.contains(&19));
    }
}