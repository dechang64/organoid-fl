use crate::blockchain::{AuditChain, OperationType};
use crate::db::VectorDB;
use crate::proto::vector_db_server::{VectorDb, VectorDbServer};
use crate::proto::{DeleteRequest, DeleteResponse, InsertRequest, InsertResponse, SearchRequest, SearchResponse, SearchResult, StatsRequest, StatsResponse};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

pub struct VectorDbService {
    db: Arc<RwLock<VectorDB>>,
    audit: Arc<RwLock<AuditChain>>,
}

impl VectorDbService {
    pub fn new(db: Arc<RwLock<VectorDB>>, audit: Arc<RwLock<AuditChain>>) -> Self {
        Self { db, audit }
    }
}

#[tonic::async_trait]
impl VectorDb for VectorDbService {
    async fn insert(
        &self, req: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let vectors = req.into_inner().vectors;
        let count = vectors.len();
        let mut db = self.db.write().await;
        let mut entries = Vec::with_capacity(vectors.len());
        for v in vectors {
            let metadata: HashMap<String, String> = v.metadata;
            entries.push(crate::db::VectorEntry {
                id: v.id,
                values: v.values,
                metadata,
            });
        }
        let inserted = db
            .insert_batch(entries)
            .map_err(|e| Status::internal(e.to_string()))? as i32;

        // Log to audit chain
        drop(db);
        let mut audit = self.audit.write().await;
        audit.append(OperationType::Insert { count });

        Ok(Response::new(InsertResponse { inserted }))
    }

    async fn search(
        &self, req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = req.into_inner();
        let query = req.query;
        let k = req.k.max(1) as usize;

        let mut db = self.db.write().await;
        let results = db.search(&query, k);

        // Log to audit chain
        let result_count = results.len();
        drop(db);
        let mut audit = self.audit.write().await;
        audit.append(OperationType::Search { k, result_count });

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, distance, metadata)| SearchResult {
                id,
                values: vec![],
                distance,
                metadata,
            })
            .collect();
        Ok(Response::new(SearchResponse { results: search_results }))
    }

    async fn delete(
        &self, req: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let ids = req.into_inner().ids;
        let count = ids.len();
        let mut db = self.db.write().await;
        let deleted = db.delete(&ids) as i32;

        // Log to audit chain
        drop(db);
        let mut audit = self.audit.write().await;
        audit.append(OperationType::Delete { count });

        Ok(Response::new(DeleteResponse { deleted }))
    }

    async fn stats(
        &self, _req: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let db = self.db.read().await;
        Ok(Response::new(StatsResponse {
            total_vectors: db.len() as i64,
            dimension: db.dimension() as i32,
            index_size: db.index_size() as i64,
        }))
    }
}

pub fn create_server(db: Arc<RwLock<VectorDB>>, audit: Arc<RwLock<AuditChain>>) -> VectorDbServer<VectorDbService> {
    VectorDbServer::new(VectorDbService::new(db, audit))
}
