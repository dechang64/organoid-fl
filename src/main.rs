use clap::Parser;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use organoid_vectordb::blockchain::AuditChain;
use organoid_vectordb::db::VectorDB;
use organoid_vectordb::grpc::create_server;
use organoid_vectordb::web::create_router;

#[derive(Parser)]
#[command(name = "organoid-vectordb", about = "Vector database for organoid image FL platform")]
struct Cli {
    /// Dimension of vectors
    #[arg(short, long, default_value = "128")]
    dimension: usize,

    /// gRPC server port
    #[arg(short, long, default_value = "50051")]
    port: u16,

    /// HTTP dashboard port
    #[arg(long, default_value = "8080")]
    http_port: u16,

    /// Max audit chain blocks to keep in memory
    #[arg(long, default_value = "1000")]
    max_audit_blocks: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let grpc_addr = format!("0.0.0.0:{}", cli.port);
    let http_addr = format!("0.0.0.0:{}", cli.http_port);

    info!("Starting Organoid VectorDB");
    info!("  Dimension: {}", cli.dimension);
    info!("  gRPC:     {}", grpc_addr);
    info!("  HTTP:     {}", http_addr);

    let db = Arc::new(RwLock::new(VectorDB::new(cli.dimension)));
    let audit = Arc::new(RwLock::new(AuditChain::new(cli.max_audit_blocks)));

    // Start gRPC server
    let grpc_db = db.clone();
    let grpc_audit = audit.clone();
    let grpc_handle = tokio::spawn(async move {
        let server = create_server(grpc_db, grpc_audit);
        info!("gRPC server ready on {}", grpc_addr);
        if let Err(e) = tonic::transport::Server::builder()
            .add_service(server)
            .serve(grpc_addr.parse().unwrap())
            .await
        {
            tracing::error!("gRPC server error: {}", e);
        }
    });

    // Start HTTP dashboard server
    let http_db = db.clone();
    let http_audit = audit.clone();
    let http_handle = tokio::spawn(async move {
        let app = create_router(http_db, http_audit);
        let listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
        info!("HTTP dashboard ready on http://{}", http_addr);
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    // Wait for either server to fail
    tokio::select! {
        _ = grpc_handle => info!("gRPC server stopped"),
        _ = http_handle => info!("HTTP server stopped"),
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    info!("Server shut down");
    Ok(())
}
