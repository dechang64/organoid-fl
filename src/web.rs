//! Web dashboard for the Organoid VectorDB platform.
//!
//! Provides a REST API and embedded HTML dashboard for monitoring:
//! - VectorDB statistics
//! - Search interface
//! - Blockchain audit log viewer
//! - FL training metrics

use crate::blockchain::{AuditChain, OperationType};
use crate::db::VectorDB;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<VectorDB>>,
    pub audit: Arc<RwLock<AuditChain>>,
}

// ============================================================
// API Types
// ============================================================

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_vectors: usize,
    pub dimension: usize,
    pub index_size: u64,
    pub hnsw_layers: usize,
    pub chain_length: usize,
    pub chain_valid: bool,
    pub latest_hash: String,
}

// ============================================================
// Handlers
// ============================================================

async fn dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn get_stats(State(state): State<AppState>) -> Json<StatsResponse> {
    let db = state.db.read().await;
    let audit = state.audit.read().await;
    Json(StatsResponse {
        total_vectors: db.len(),
        dimension: db.dimension(),
        index_size: db.index_size(),
        hnsw_layers: db.hnsw_layers(),
        chain_length: audit.len(),
        chain_valid: audit.verify_chain(),
        latest_hash: audit.latest_hash().to_string(),
    })
}

async fn post_search(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let start = std::time::Instant::now();

    let query: Vec<f32> = match body.get("query") {
        Some(q) => match serde_json::from_value(q.clone()) {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))),
        },
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "missing 'query' field"}))),
    };

    let k: usize = body
        .get("k")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(10);

    let mut db = state.db.write().await;
    let results = db.search(&query, k);
    let elapsed = start.elapsed();

    // Log to audit chain
    let result_count = results.len();
    drop(db);
    let mut audit = state.audit.write().await;
    audit.append(OperationType::Search { k, result_count });

    let results_json: Vec<serde_json::Value> = results
        .into_iter()
        .map(|(id, distance, metadata)| {
            serde_json::json!({
                "id": id,
                "distance": distance,
                "metadata": metadata,
            })
        })
        .collect();

    (StatusCode::OK, Json(serde_json::json!({
        "results": results_json,
        "query_time_ms": elapsed.as_secs_f64() * 1000.0,
        "k": k,
    })))
}

async fn post_insert(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let vectors = match body.get("vectors") {
        Some(v) => match serde_json::from_value::<Vec<serde_json::Value>>(v.clone()) {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))),
        },
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "missing 'vectors' field"}))),
    };

    let mut entries = Vec::new();
    for v in vectors {
        let id = v.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let values: Vec<f32> = match v.get("values") {
            Some(val) => match serde_json::from_value(val.clone()) {
                Ok(v) => v,
                Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))),
            },
            None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "missing 'values' in vector"}))),
        };
        let metadata: std::collections::HashMap<String, String> = v
            .get("metadata")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();
        entries.push(crate::db::VectorEntry { id, values, metadata });
    }

    let count = entries.len();
    let mut db = state.db.write().await;
    let inserted = match db.insert_batch(entries) {
        Ok(n) => n,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))),
    };

    // Log to audit chain
    drop(db);
    let mut audit = state.audit.write().await;
    audit.append(OperationType::Insert { count: inserted });

    (StatusCode::OK, Json(serde_json::json!({
        "inserted": inserted,
        "requested": count,
    })))
}

async fn get_audit(State(state): State<AppState>) -> Json<serde_json::Value> {
    let audit = state.audit.read().await;
    let blocks: Vec<serde_json::Value> = audit
        .recent(100)
        .iter()
        .map(|b| {
            serde_json::json!({
                "index": b.index,
                "timestamp": b.timestamp,
                "operation": format!("{:?}", b.operation),
                "hash": b.hash,
                "prev_hash": b.prev_hash,
            })
        })
        .collect();
    Json(serde_json::json!({
        "blocks": blocks,
        "total": audit.len(),
        "valid": audit.verify_chain(),
    }))
}

async fn get_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "organoid-vectordb",
    }))
}

// ============================================================
// Router
// ============================================================

pub fn create_router(db: Arc<RwLock<VectorDB>>, audit: Arc<RwLock<AuditChain>>) -> Router {
    let state = AppState { db, audit };
    Router::new()
        .route("/", get(dashboard))
        .route("/api/health", get(get_health))
        .route("/api/stats", get(get_stats))
        .route("/api/search", post(post_search))
        .route("/api/insert", post(post_insert))
        .route("/api/audit", get(get_audit))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ============================================================
// Dashboard HTML
// ============================================================

const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Organoid VectorDB Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-bottom: 1px solid #334155;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 1.25rem;
            font-weight: 600;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: #94a3b8;
        }
        .header .status .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
        .tabs {
            display: flex;
            gap: 0.25rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #334155;
            padding-bottom: 0;
        }
        .tab {
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            border: none;
            background: none;
            color: #94a3b8;
            font-size: 0.875rem;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab:hover { color: #e2e8f0; }
        .tab.active {
            color: #38bdf8;
            border-bottom-color: #38bdf8;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 1.25rem;
        }
        .stat-card .label {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .stat-card .value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .stat-card .value.accent { color: #38bdf8; }
        .stat-card .value.green { color: #22c55e; }
        .stat-card .value.purple { color: #a78bfa; }
        .stat-card .value.amber { color: #fbbf24; }
        .panel {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .panel h3 {
            font-size: 0.875rem;
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .search-box {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .search-box input, .search-box textarea {
            flex: 1;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            padding: 0.625rem 0.875rem;
            color: #e2e8f0;
            font-size: 0.875rem;
            font-family: 'SF Mono', 'Fira Code', monospace;
        }
        .search-box input:focus, .search-box textarea:focus {
            outline: none;
            border-color: #38bdf8;
        }
        .search-box textarea {
            min-height: 60px;
            resize: vertical;
        }
        .btn {
            padding: 0.625rem 1.25rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #2563eb;
            color: white;
        }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-secondary {
            background: #334155;
            color: #e2e8f0;
        }
        .btn-secondary:hover { background: #475569; }
        .results-table {
            width: 100%;
            border-collapse: collapse;
        }
        .results-table th {
            text-align: left;
            padding: 0.5rem 0.75rem;
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #334155;
        }
        .results-table td {
            padding: 0.5rem 0.75rem;
            font-size: 0.875rem;
            border-bottom: 1px solid #1e293b;
        }
        .results-table tr:hover td { background: #0f172a; }
        .audit-chain {
            max-height: 600px;
            overflow-y: auto;
        }
        .audit-block {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.8125rem;
            font-family: 'SF Mono', 'Fira Code', monospace;
        }
        .audit-block .block-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        }
        .audit-block .block-idx {
            color: #38bdf8;
            font-weight: 600;
        }
        .audit-block .block-time {
            color: #64748b;
        }
        .audit-block .block-op {
            color: #a78bfa;
            margin-bottom: 0.25rem;
        }
        .audit-block .block-hash {
            color: #475569;
            font-size: 0.75rem;
        }
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: #64748b;
        }
        .chain-status {
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.25rem 0.625rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .chain-valid {
            background: rgba(34, 197, 94, 0.1);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        .chain-invalid {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        .query-time {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        .query-time span { color: #22c55e; font-weight: 600; }
        .insert-section { margin-top: 1rem; }
        .insert-section h4 {
            font-size: 0.8125rem;
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Organoid VectorDB</h1>
        <div class="status">
            <div class="dot"></div>
            <span id="statusText">Connected</span>
        </div>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="switchTab('overview')">Overview</button>
            <button class="tab" onclick="switchTab('search')">Search</button>
            <button class="tab" onclick="switchTab('audit')">Audit Chain</button>
        </div>

        <!-- Overview Tab -->
        <div id="tab-overview" class="tab-content active">
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="label">Total Vectors</div>
                    <div class="value accent" id="statVectors">-</div>
                </div>
                <div class="stat-card">
                    <div class="label">Dimension</div>
                    <div class="value" id="statDimension">-</div>
                </div>
                <div class="stat-card">
                    <div class="label">HNSW Layers</div>
                    <div class="value purple" id="statLayers">-</div>
                </div>
                <div class="stat-card">
                    <div class="label">Index Size</div>
                    <div class="value green" id="statIndex">-</div>
                </div>
                <div class="stat-card">
                    <div class="label">Chain Length</div>
                    <div class="value amber" id="statChain">-</div>
                </div>
                <div class="stat-card">
                    <div class="label">Chain Integrity</div>
                    <div id="statChainValid">-</div>
                </div>
            </div>
            <div class="panel">
                <h3>System Info</h3>
                <div style="font-size: 0.875rem; color: #94a3b8; line-height: 1.75;">
                    <div><strong style="color: #e2e8f0;">Engine:</strong> HNSW (Hierarchical Navigable Small World)</div>
                    <div><strong style="color: #e2e8f0;">Metric:</strong> Euclidean Distance</div>
                    <div><strong style="color: #e2e8f0;">M:</strong> 16 (max connections per layer)</div>
                    <div><strong style="color: #e2e8f0;">M0:</strong> 32 (max connections at layer 0)</div>
                    <div><strong style="color: #e2e8f0;">Audit:</strong> SHA-256 hash chain</div>
                </div>
            </div>
        </div>

        <!-- Search Tab -->
        <div id="tab-search" class="tab-content">
            <div class="panel">
                <h3>Nearest Neighbor Search</h3>
                <div class="search-box">
                    <textarea id="queryInput" placeholder="Enter query vector, e.g. [0.1, 0.2, 0.3, ...]"></textarea>
                    <input type="number" id="kInput" value="10" min="1" max="100" style="width: 80px;" placeholder="k">
                    <button class="btn btn-primary" onclick="doSearch()">Search</button>
                </div>
                <div id="searchResults">
                    <div class="empty-state">Enter a query vector to search</div>
                </div>
            </div>
            <div class="panel insert-section">
                <h3>Insert Vectors</h3>
                <div class="search-box">
                    <textarea id="insertInput" placeholder='[{"id": "v1", "values": [0.1, 0.2, ...], "metadata": {"class": "early"}}]'></textarea>
                    <button class="btn btn-secondary" onclick="doInsert()">Insert</button>
                </div>
                <div id="insertResult"></div>
            </div>
        </div>

        <!-- Audit Tab -->
        <div id="tab-audit" class="tab-content">
            <div class="panel">
                <h3>Blockchain Audit Log</h3>
                <div class="audit-chain" id="auditLog">
                    <div class="empty-state">Loading...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function refreshStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('statVectors').textContent = data.total_vectors.toLocaleString();
                document.getElementById('statDimension').textContent = data.dimension;
                document.getElementById('statLayers').textContent = data.hnsw_layers;
                document.getElementById('statIndex').textContent = data.index_size.toLocaleString();
                document.getElementById('statChain').textContent = data.chain_length.toLocaleString();
                const validEl = document.getElementById('statChainValid');
                if (data.chain_valid) {
                    validEl.innerHTML = '<span class="chain-status chain-valid">✓ Valid</span>';
                } else {
                    validEl.innerHTML = '<span class="chain-status chain-invalid">✗ Tampered</span>';
                }
            } catch (e) {
                document.getElementById('statusText').textContent = 'Disconnected';
            }
        }

        async function doSearch() {
            const queryStr = document.getElementById('queryInput').value.trim();
            if (!queryStr) return;
            try {
                const query = JSON.parse(queryStr);
                const k = parseInt(document.getElementById('kInput').value) || 10;
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, k }),
                });
                const data = await res.json();
                if (data.error) {
                    document.getElementById('searchResults').innerHTML = `<div class="empty-state">${data.error}</div>`;
                    return;
                }
                let html = `<table class="results-table">
                    <thead><tr><th>#</th><th>ID</th><th>Distance</th><th>Metadata</th></tr></thead><tbody>`;
                data.results.forEach((r, i) => {
                    const meta = Object.entries(r.metadata || {}).map(([k,v]) => `${k}=${v}`).join(', ');
                    html += `<tr><td>${i+1}</td><td>${escapeHtml(r.id)}</td><td>${r.distance.toFixed(6)}</td><td>${escapeHtml(meta)}</td></tr>`;
                });
                html += '</tbody></table>';
                html += `<div class="query-time">Query time: <span>${data.query_time_ms.toFixed(2)} ms</span></div>`;
                document.getElementById('searchResults').innerHTML = html;
            } catch (e) {
                document.getElementById('searchResults').innerHTML = `<div class="empty-state">Invalid JSON: ${e.message}</div>`;
            }
        }

        async function doInsert() {
            const input = document.getElementById('insertInput').value.trim();
            if (!input) return;
            try {
                const vectors = JSON.parse(input);
                const res = await fetch('/api/insert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ vectors }),
                });
                const data = await res.json();
                if (data.error) {
                    document.getElementById('insertResult').innerHTML = `<div class="empty-state">${data.error}</div>`;
                } else {
                    document.getElementById('insertResult').innerHTML = `<div style="color: #22c55e; font-size: 0.875rem;">Inserted ${data.inserted}/${data.requested} vectors</div>`;
                    refreshStats();
                }
            } catch (e) {
                document.getElementById('insertResult').innerHTML = `<div class="empty-state">Invalid JSON: ${e.message}</div>`;
            }
        }

        async function loadAudit() {
            try {
                const res = await fetch('/api/audit');
                const data = await res.json();
                let html = '';
                if (data.blocks.length === 0) {
                    html = '<div class="empty-state">No audit entries yet</div>';
                } else {
                    data.blocks.forEach(b => {
                        html += `<div class="audit-block">
                            <div class="block-header">
                                <span class="block-idx">#${b.index}</span>
                                <span class="block-time">${new Date(b.timestamp).toLocaleString()}</span>
                            </div>
                            <div class="block-op">${escapeHtml(b.operation)}</div>
                            <div class="block-hash">hash: ${b.hash.substring(0, 16)}... → prev: ${b.prev_hash.substring(0, 16)}...</div>
                        </div>`;
                    });
                }
                document.getElementById('auditLog').innerHTML = html;
            } catch (e) {
                document.getElementById('auditLog').innerHTML = `<div class="empty-state">Error loading audit</div>`;
            }
        }

        function switchTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab[onclick="switchTab('${name}')"]`).classList.add('active');
            document.getElementById('tab-' + name).classList.add('active');
        }

        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        // Initial load
        refreshStats();
        loadAudit();
        setInterval(refreshStats, 5000);
        setInterval(loadAudit, 10000);
    </script>
</body>
</html>"##;
