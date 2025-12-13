//! Knowledge graph IPC commands.
//!
//! Handles PKG visualization data retrieval.
//! Reads parsed documents from workspace and converts to graph format.
//!
//! IMPORTANT: Uses document aggregation to prevent "thousands of nodes" issue.
//! Unstructured.io creates 60-100+ elements per document (each paragraph = element).
//! This module aggregates elements by their source document (parent_id/filename).

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tauri::command;

/// Regex for extracting markdown headings (# Title)
static HEADING_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_heading_regex() -> &'static Regex {
    HEADING_REGEX.get_or_init(|| {
        Regex::new(r"(?m)^#\s+(.+)$").expect("Invalid heading regex")
    })
}

/// Extract title from YAML frontmatter in metadata.extra
fn extract_frontmatter_title(extra: &HashMap<String, serde_json::Value>) -> Option<String> {
    // Check for direct title field
    if let Some(title) = extra.get("title") {
        if let Some(t) = title.as_str() {
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    // Check for frontmatter object with title
    if let Some(frontmatter) = extra.get("frontmatter") {
        if let Some(obj) = frontmatter.as_object() {
            if let Some(title) = obj.get("title") {
                if let Some(t) = title.as_str() {
                    if !t.is_empty() {
                        return Some(t.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Extract first markdown heading from text
fn extract_markdown_heading(text: &str) -> Option<String> {
    let regex = get_heading_regex();
    if let Some(caps) = regex.captures(text) {
        if let Some(heading) = caps.get(1) {
            let h = heading.as_str().trim();
            if !h.is_empty() {
                return Some(h.to_string());
            }
        }
    }
    None
}

/// Clean filename by removing extension and path prefix
fn clean_filename(filename: &str) -> String {
    let name = filename
        .rsplit('/')
        .next()
        .unwrap_or(filename);

    // Remove common extensions
    let cleaned = name
        .trim_end_matches(".json")
        .trim_end_matches(".md")
        .trim_end_matches(".txt")
        .trim_end_matches(".pdf");

    // Replace underscores/dashes with spaces for readability
    cleaned.replace('_', " ").replace('-', " ")
}

/// Truncate string to max length with ellipsis
fn truncate_label(text: &str, max_len: usize) -> String {
    if text.chars().count() > max_len {
        format!("{}...", text.chars().take(max_len).collect::<String>())
    } else {
        text.to_string()
    }
}

/// Graph node for visualization.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub timestamp: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Graph edge/link for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphLink {
    pub source: String,
    pub target: String,
    pub relationship: String,
    pub weight: Option<f64>,
    /// Confidence score for this relationship (0.0 - 1.0)
    pub confidence: Option<f64>,
}

/// Complete graph data for react-force-graph.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
    /// Pagination metadata
    pub pagination: Option<PaginationMeta>,
}

/// Pagination metadata for graph queries.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PaginationMeta {
    /// Current offset in the result set
    pub offset: u32,
    /// Total number of nodes available
    pub total: u32,
    /// Whether more results are available
    pub has_more: bool,
}

/// Filter parameters for subgraph queries.
#[derive(Debug, Deserialize)]
pub struct GraphFilter {
    /// Filter by source identifiers
    pub sources: Option<Vec<String>>,
    /// Filter by node types (entity types)
    pub node_types: Option<Vec<String>>,
    /// Minimum confidence threshold (0.0 - 1.0)
    pub min_confidence: Option<f64>,
    /// Start of time range (ISO 8601)
    pub start_date: Option<String>,
    /// End of time range (ISO 8601)
    pub end_date: Option<String>,
}

/// Graph statistics response.
#[derive(Debug, Serialize)]
pub struct GraphStats {
    /// Total number of nodes
    pub total_nodes: u32,
    /// Total number of links
    pub total_links: u32,
    /// Count of nodes by type
    pub nodes_by_type: HashMap<String, u32>,
    /// Count of nodes by source
    pub nodes_by_source: HashMap<String, u32>,
}

/// Key for grouping elements by source document.
/// Used to aggregate multiple Unstructured.io elements into a single graph node.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
enum DocumentKey {
    /// IMAP email identified by source + uid
    ImapEmail { source: String, uid: u64 },
    /// Unstructured element grouped by parent_id or filename
    UnstructuredDoc { parent_id: String },
    /// Raw email identified by sha256 hash
    RawEmail { sha256: String },
}

/// Aggregated document from multiple parsed elements.
/// Combines all elements from the same source document into one graph node.
#[derive(Debug)]
struct AggregatedDocument {
    /// Unique identifier for the aggregated document
    id: String,
    /// Display label (filename, subject, or first element text)
    label: String,
    /// Node type (Document, Email, Code, etc.)
    node_type: String,
    /// Timestamp from earliest/latest element
    timestamp: Option<String>,
    /// Source identifier for linking
    source: Option<String>,
    /// Count of elements in this document
    element_count: usize,
    /// Types of elements contained (Title, NarrativeText, etc.)
    element_types: Vec<String>,
    /// Total character count across all elements
    total_chars: usize,
    /// Additional metadata (subject, sender, etc. for emails)
    metadata: serde_json::Value,
}

impl Default for AggregatedDocument {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            node_type: "Document".to_string(),
            timestamp: None,
            source: None,
            element_count: 0,
            element_types: Vec::new(),
            total_chars: 0,
            metadata: serde_json::Value::Null,
        }
    }
}

/// Parsed element from Unstructured.io output.
#[derive(Debug, Deserialize)]
struct ParsedElement {
    #[serde(rename = "type")]
    element_type: String,
    element_id: String,
    text: String,
    metadata: ParsedMetadata,
}

/// Metadata from parsed element.
#[derive(Debug, Deserialize)]
struct ParsedMetadata {
    filename: Option<String>,
    source: Option<String>,
    ingested_at: Option<String>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

/// Extracted entity from Python entity extractor.
#[derive(Debug, Deserialize)]
struct ExtractedEntity {
    id: String,
    name: String,
    entity_type: String,
    #[serde(default)]
    canonical_name: Option<String>,
    #[serde(default)]
    confidence: f64,
    #[serde(default)]
    extraction_method: Option<String>,
}

/// Relationship between document and entity.
#[derive(Debug, Deserialize)]
struct ExtractedRelationship {
    source: String,
    target: String,
    relationship: String,
    #[serde(default)]
    confidence: f64,
}

/// Document entities JSON format from Python entity extractor.
#[derive(Debug, Deserialize)]
struct DocumentEntities {
    document_id: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    entities: Vec<ExtractedEntity>,
    #[serde(default)]
    relationships: Vec<ExtractedRelationship>,
}

/// IMAP email format from fast sync.
#[derive(Debug, Deserialize)]
struct ImapEmail {
    sha256: String,
    content: String,
    metadata: ImapMetadata,
}

/// Metadata from IMAP email.
#[derive(Debug, Deserialize)]
struct ImapMetadata {
    source_id: Option<String>,
    source_type: Option<String>,
    source: Option<String>,
    uid: Option<u64>,
    message_id: Option<String>,
    subject: Option<String>,
    sender: Option<String>,
    recipient: Option<String>,
    folder: Option<String>,
    date: Option<String>,
    #[serde(rename = "extractionTimestamp")]
    extraction_timestamp: Option<String>,
}

/// Get workspace path: ~/.futurnal/workspace
fn get_workspace_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(format!("{home}/.futurnal/workspace"))
}

/// Load all entity files from workspace/entities/
/// Returns: (entity_nodes, entity_links, entity_ids, document_titles)
fn load_entity_files(
    workspace: &PathBuf,
) -> (Vec<GraphNode>, Vec<GraphLink>, HashMap<String, String>, HashMap<String, String>) {
    let entities_dir = workspace.join("entities");
    let mut entity_nodes: Vec<GraphNode> = Vec::new();
    let mut entity_links: Vec<GraphLink> = Vec::new();
    let mut entity_node_ids: HashMap<String, String> = HashMap::new(); // entity_id -> display name
    let mut document_titles: HashMap<String, String> = HashMap::new(); // document_id -> title

    if !entities_dir.exists() {
        return (entity_nodes, entity_links, entity_node_ids, document_titles);
    }

    let entries = match std::fs::read_dir(&entities_dir) {
        Ok(e) => e,
        Err(_) => return (entity_nodes, entity_links, entity_node_ids, document_titles),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.extension().map_or(false, |ext| ext == "json") {
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let doc_entities: DocumentEntities = match serde_json::from_str(&content) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Store document title if present (for updating document node labels)
        if let Some(title) = doc_entities.title {
            if !title.is_empty() {
                document_titles.insert(doc_entities.document_id.clone(), title);
            }
        }

        // Create entity nodes (deduplicated by entity_id)
        for entity in doc_entities.entities {
            if !entity_node_ids.contains_key(&entity.id) {
                entity_node_ids.insert(entity.id.clone(), entity.name.clone());

                entity_nodes.push(GraphNode {
                    id: entity.id.clone(),
                    label: entity.canonical_name.unwrap_or(entity.name.clone()),
                    node_type: entity.entity_type,
                    timestamp: None,
                    metadata: Some(serde_json::json!({
                        "extraction_method": entity.extraction_method,
                        "confidence": entity.confidence,
                    })),
                });
            }
        }

        // Create relationship links (document → entity)
        // Note: rel.source is the document hash, but document nodes use "doc:{hash}" format
        for rel in doc_entities.relationships {
            entity_links.push(GraphLink {
                source: format!("doc:{}", rel.source),
                target: rel.target,
                relationship: rel.relationship,
                weight: Some(rel.confidence),
                confidence: Some(rel.confidence),
            });
        }
    }

    log::info!(
        "Loaded {} entity nodes, {} relationships, {} document titles from entities/",
        entity_nodes.len(),
        entity_links.len(),
        document_titles.len()
    );

    (entity_nodes, entity_links, entity_node_ids, document_titles)
}

/// Map Unstructured.io element types to our EntityType.
/// See: https://docs.unstructured.io/api-reference/api-services/document-elements
fn map_element_type_to_entity(element_type: &str) -> &'static str {
    match element_type {
        // Text content types → Document
        "NarrativeText" | "Text" | "UncategorizedText" | "Title" | "Header" | "Footer" => "Document",
        // List types → Document
        "ListItem" | "List" => "Document",
        // Tables → Document
        "Table" | "TableChunk" => "Document",
        // Code blocks → Code
        "CodeSnippet" => "Code",
        // Media → Document
        "Image" | "Figure" | "FigureCaption" => "Document",
        // Formula → Concept
        "Formula" => "Concept",
        // Address/contact → Person
        "Address" => "Person",
        // Page elements → Document
        "PageBreak" | "PageNumber" => "Document",
        // Default fallback
        _ => "Document",
    }
}

/// Extract document key from a parsed element for aggregation.
/// Groups elements by their source document using parent_id, uid, or filename.
fn get_document_key(element: &ParsedElement) -> DocumentKey {
    // Try parent_id first (Unstructured.io sets this for chunked documents)
    if let Some(parent_id) = element.metadata.extra.get("parent_id") {
        if let Some(pid) = parent_id.as_str() {
            if !pid.is_empty() {
                return DocumentKey::UnstructuredDoc { parent_id: pid.to_string() };
            }
        }
    }

    // Try uid + source for IMAP elements
    if let (Some(uid), Some(source)) = (
        element.metadata.extra.get("uid").and_then(|v| v.as_u64()),
        element.metadata.source.as_ref()
    ) {
        return DocumentKey::ImapEmail { source: source.clone(), uid };
    }

    // Fall back to filename for local files
    let fallback = element.metadata.filename.clone()
        .unwrap_or_else(|| element.element_id.clone());
    DocumentKey::UnstructuredDoc { parent_id: fallback }
}

/// Aggregate an element into the documents map.
/// Updates the aggregated document with element data.
///
/// Title extraction priority:
/// 1. YAML frontmatter title field
/// 2. Unstructured.io "Title" element
/// 3. First markdown heading (# Heading)
/// 4. Clean filename (without extension)
fn aggregate_element(
    documents: &mut HashMap<DocumentKey, AggregatedDocument>,
    key: DocumentKey,
    element: &ParsedElement,
) {
    let doc = documents.entry(key.clone()).or_insert_with(|| {
        let id = match &key {
            DocumentKey::ImapEmail { source, uid } => format!("email:{}:{}", source, uid),
            DocumentKey::UnstructuredDoc { parent_id } => format!("doc:{}", parent_id),
            DocumentKey::RawEmail { sha256 } => sha256.clone(),
        };

        // Try to extract title from frontmatter first
        let initial_label = extract_frontmatter_title(&element.metadata.extra)
            .map(|t| truncate_label(&t, 80))
            .unwrap_or_else(|| {
                // Fall back to clean filename
                element.metadata.filename.as_ref()
                    .map(|f| clean_filename(f))
                    .unwrap_or_else(|| truncate_label(&element.text, 50))
            });

        let path_value = element.metadata.extra.get("path");

        AggregatedDocument {
            id,
            label: initial_label,
            node_type: map_element_type_to_entity(&element.element_type).to_string(),
            source: element.metadata.source.clone(),
            metadata: serde_json::json!({
                "filename": element.metadata.filename,
                "source": element.metadata.source,
                "path": path_value,
            }),
            ..Default::default()
        }
    });

    // Update aggregated stats
    doc.element_count += 1;
    doc.total_chars += element.text.len();

    // Track element types
    if !doc.element_types.contains(&element.element_type) {
        doc.element_types.push(element.element_type.clone());
    }

    // Title extraction priority: frontmatter > Title element > markdown heading
    // Check if we already have a good title from frontmatter
    let has_frontmatter_title = extract_frontmatter_title(&element.metadata.extra).is_some();

    if !has_frontmatter_title {
        // Try Title element type (from Unstructured.io)
        if element.element_type == "Title" && !element.text.is_empty() {
            doc.label = truncate_label(&element.text, 80);
        }
        // Try markdown heading from NarrativeText if no Title element found yet
        else if element.element_type == "NarrativeText"
            && !doc.element_types.contains(&"Title".to_string())
        {
            if let Some(heading) = extract_markdown_heading(&element.text) {
                // Only update if current label looks like a filename
                if doc.label.contains('.') || doc.label.contains(' ') && doc.label.len() < 30 {
                    doc.label = truncate_label(&heading, 80);
                }
            }
        }
    }

    // Update timestamp if newer
    if let Some(ts) = &element.metadata.ingested_at {
        if doc.timestamp.is_none() || doc.timestamp.as_ref().unwrap() < ts {
            doc.timestamp = Some(ts.clone());
        }
    }

    // Update source if not set
    if doc.source.is_none() {
        doc.source = element.metadata.source.clone();
    }
}

/// Get knowledge graph data for visualization.
///
/// Reads parsed documents from workspace/parsed/ directory
/// and converts them to graph nodes.
///
/// IMPORTANT: Uses aggregation to prevent "thousands of nodes" issue.
/// Unstructured.io elements are grouped by their source document (parent_id/filename).
/// Raw IMAP emails are kept as individual nodes (already 1 per email).
#[command]
pub async fn get_knowledge_graph(limit: Option<u32>) -> Result<GraphData, String> {
    let limit = limit.unwrap_or(1000) as usize;
    log::info!("Getting knowledge graph (limit: {})", limit);

    let parsed_dir = get_workspace_path().join("parsed");

    // HashMap to aggregate documents by their key
    let mut documents: HashMap<DocumentKey, AggregatedDocument> = HashMap::new();
    let mut source_nodes: HashMap<String, String> = HashMap::new();
    let mut files_processed = 0;

    // Read parsed files and aggregate by source document
    if parsed_dir.exists() {
        let entries = std::fs::read_dir(&parsed_dir)
            .map_err(|e| format!("Failed to read parsed directory: {}", e))?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if !path.extension().map_or(false, |ext| ext == "json") {
                continue;
            }

            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            files_processed += 1;

            // Try parsing as Unstructured.io format first
            if let Ok(element) = serde_json::from_str::<ParsedElement>(&content) {
                // Aggregate elements by their source document
                let key = get_document_key(&element);
                aggregate_element(&mut documents, key, &element);
            }
            // Try parsing as raw IMAP email format (already 1 per email, no aggregation needed)
            else if let Ok(email) = serde_json::from_str::<ImapEmail>(&content) {
                let key = DocumentKey::RawEmail { sha256: email.sha256.clone() };

                // Insert raw email as its own document
                documents.entry(key).or_insert_with(|| {
                    // Use subject as label, or first 50 chars of content
                    let label = email.metadata.subject.clone()
                        .map(|s| truncate_label(&s, 80))
                        .unwrap_or_else(|| truncate_label(&email.content, 50));

                    AggregatedDocument {
                        id: email.sha256.clone(),
                        label,
                        node_type: "Email".to_string(),
                        timestamp: email.metadata.date.clone(),
                        source: email.metadata.source.clone(),
                        element_count: 1,
                        element_types: vec!["RawEmail".to_string()],
                        total_chars: email.content.len(),
                        metadata: serde_json::json!({
                            "subject": email.metadata.subject,
                            "sender": email.metadata.sender,
                            "recipient": email.metadata.recipient,
                            "folder": email.metadata.folder,
                            "source": email.metadata.source,
                            "message_id": email.metadata.message_id,
                            "text": truncate_label(&email.content, 500),
                        }),
                    }
                });
            }
        }
    }

    log::info!(
        "Processed {} files, aggregated into {} documents",
        files_processed,
        documents.len()
    );

    // Convert aggregated documents to graph nodes (apply limit here)
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut links: Vec<GraphLink> = Vec::new();

    for (_key, doc) in documents.into_iter().take(limit) {
        // Determine source node type BEFORE moving doc.node_type
        let is_email = doc.node_type == "Email";

        // Create node for this aggregated document
        nodes.push(GraphNode {
            id: doc.id.clone(),
            label: doc.label,
            node_type: doc.node_type,
            timestamp: doc.timestamp,
            metadata: Some(serde_json::json!({
                "element_count": doc.element_count,
                "element_types": doc.element_types,
                "total_chars": doc.total_chars,
                "source": doc.source,
                "details": doc.metadata,
            })),
        });

        // Create source node and link if source exists
        if let Some(source) = &doc.source {
            let source_id = format!("source:{}", source);

            if !source_nodes.contains_key(&source_id) {
                source_nodes.insert(source_id.clone(), source.clone());

                // Determine source node type based on document type
                let source_type = if is_email { "Mailbox" } else { "Source" };

                nodes.push(GraphNode {
                    id: source_id.clone(),
                    label: source.clone(),
                    node_type: source_type.to_string(),
                    timestamp: None,
                    metadata: None,
                });
            }

            links.push(GraphLink {
                source: source_id,
                target: doc.id,
                relationship: "contains".to_string(),
                weight: Some(1.0),
                confidence: Some(1.0),
            });
        }
    }

    // Load entity nodes and relationship links from entities/ directory
    let workspace = get_workspace_path();
    let (entity_nodes, entity_links, _entity_ids, document_titles) = load_entity_files(&workspace);

    // Update document node labels with extracted titles (if available)
    for node in &mut nodes {
        if let Some(title) = document_titles.get(&node.id) {
            // Only update if current label looks like a raw filename (has hash-like chars or extension)
            if node.label.len() > 30 || node.label.contains('.') || !node.label.contains(' ') {
                node.label = truncate_label(title, 80);
            }
        }
    }

    // Add entity nodes to the graph
    nodes.extend(entity_nodes);

    // Add entity relationship links (filter to only include links where both source and target exist)
    let node_ids: std::collections::HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
    for link in entity_links {
        // Check if both source and target nodes exist in the graph
        if node_ids.contains(&link.source) && node_ids.contains(&link.target) {
            links.push(link);
        }
    }

    let total_nodes = nodes.len() as u32;
    let has_more = total_nodes >= limit as u32;

    log::info!(
        "Returning graph with {} nodes and {} links (from {} source documents)",
        nodes.len(),
        links.len(),
        nodes.len().saturating_sub(source_nodes.len())
    );

    Ok(GraphData {
        nodes,
        links,
        pagination: Some(PaginationMeta {
            offset: 0,
            total: total_nodes,
            has_more,
        }),
    })
}

/// Open a file with the system's default application.
///
/// This is a simple wrapper around the system's file opener.
/// On macOS, it uses the `open` command.
#[command]
pub async fn open_file(path: String) -> Result<(), String> {
    use std::process::Command;

    // Validate path exists and is a file
    let path_buf = PathBuf::from(&path);
    if !path_buf.exists() {
        return Err(format!("File not found: {}", path));
    }

    log::info!("Opening file: {}", path);

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", &path])
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }

    Ok(())
}

/// Get filtered subgraph based on filter parameters.
///
/// Filters nodes by source, type, confidence, and time range.
/// Returns a subset of the full knowledge graph.
#[command]
pub async fn get_filtered_graph(
    filter: GraphFilter,
    limit: Option<u32>,
    offset: Option<u32>,
) -> Result<GraphData, String> {
    log::info!("Getting filtered graph with filter: {:?}", filter);

    // Get the full graph first
    let full_graph = get_knowledge_graph(Some(10000)).await?;

    let offset = offset.unwrap_or(0) as usize;
    let limit = limit.unwrap_or(1000) as usize;

    // Filter nodes
    let filtered_nodes: Vec<GraphNode> = full_graph.nodes.into_iter()
        .filter(|node| {
            // Filter by node type
            if let Some(ref types) = filter.node_types {
                if !types.is_empty() && !types.contains(&node.node_type) {
                    return false;
                }
            }

            // Filter by source
            if let Some(ref sources) = filter.sources {
                if !sources.is_empty() {
                    let node_source = node.metadata.as_ref()
                        .and_then(|m| m.get("source"))
                        .and_then(|s| s.as_str());

                    if let Some(ns) = node_source {
                        if !sources.iter().any(|s| ns.contains(s)) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }

            // Filter by time range
            if let Some(ref start) = filter.start_date {
                if let Some(ref ts) = node.timestamp {
                    if ts < start {
                        return false;
                    }
                }
            }
            if let Some(ref end) = filter.end_date {
                if let Some(ref ts) = node.timestamp {
                    if ts > end {
                        return false;
                    }
                }
            }

            true
        })
        .collect();

    // Build set of filtered node IDs for link filtering
    let node_ids: std::collections::HashSet<String> = filtered_nodes.iter()
        .map(|n| n.id.clone())
        .collect();

    // Filter links to only include those between filtered nodes
    let filtered_links: Vec<GraphLink> = full_graph.links.into_iter()
        .filter(|link| {
            // Check both source and target exist in filtered nodes
            if !node_ids.contains(&link.source) || !node_ids.contains(&link.target) {
                return false;
            }

            // Filter by confidence
            if let Some(min_conf) = filter.min_confidence {
                if let Some(conf) = link.confidence {
                    if conf < min_conf {
                        return false;
                    }
                }
            }

            true
        })
        .collect();

    // Apply pagination
    let total = filtered_nodes.len() as u32;
    let paginated_nodes: Vec<GraphNode> = filtered_nodes.into_iter()
        .skip(offset)
        .take(limit)
        .collect();

    let has_more = (offset + paginated_nodes.len()) < total as usize;

    Ok(GraphData {
        nodes: paginated_nodes,
        links: filtered_links,
        pagination: Some(PaginationMeta {
            offset: offset as u32,
            total,
            has_more,
        }),
    })
}

/// Get neighbors of a specific node up to a certain depth.
///
/// Returns all nodes connected to the specified node within k hops.
#[command]
pub async fn get_node_neighbors(
    node_id: String,
    depth: Option<u32>,
) -> Result<GraphData, String> {
    log::info!("Getting neighbors for node {} with depth {:?}", node_id, depth);

    let depth = depth.unwrap_or(1) as usize;

    // Get the full graph
    let full_graph = get_knowledge_graph(Some(10000)).await?;

    // Build adjacency list
    let mut adjacency: HashMap<String, Vec<(String, GraphLink)>> = HashMap::new();
    for link in &full_graph.links {
        adjacency.entry(link.source.clone())
            .or_default()
            .push((link.target.clone(), link.clone()));
        adjacency.entry(link.target.clone())
            .or_default()
            .push((link.source.clone(), link.clone()));
    }

    // BFS to find neighbors within depth
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue: Vec<(String, usize)> = vec![(node_id.clone(), 0)];
    let mut neighbor_ids: Vec<String> = Vec::new();

    while let Some((current_id, current_depth)) = queue.pop() {
        if visited.contains(&current_id) {
            continue;
        }
        visited.insert(current_id.clone());
        neighbor_ids.push(current_id.clone());

        if current_depth < depth {
            if let Some(neighbors) = adjacency.get(&current_id) {
                for (neighbor_id, _) in neighbors {
                    if !visited.contains(neighbor_id) {
                        queue.push((neighbor_id.clone(), current_depth + 1));
                    }
                }
            }
        }
    }

    // Filter nodes and links to neighborhood
    let neighbor_set: std::collections::HashSet<String> = neighbor_ids.iter().cloned().collect();

    let nodes: Vec<GraphNode> = full_graph.nodes.into_iter()
        .filter(|n| neighbor_set.contains(&n.id))
        .collect();

    let links: Vec<GraphLink> = full_graph.links.into_iter()
        .filter(|l| neighbor_set.contains(&l.source) && neighbor_set.contains(&l.target))
        .collect();

    let total = nodes.len() as u32;

    Ok(GraphData {
        nodes,
        links,
        pagination: Some(PaginationMeta {
            offset: 0,
            total,
            has_more: false,
        }),
    })
}

/// Get statistics about the knowledge graph.
///
/// Returns counts of nodes by type and source.
#[command]
pub async fn get_graph_stats() -> Result<GraphStats, String> {
    log::info!("Getting graph statistics");

    let graph = get_knowledge_graph(Some(10000)).await?;

    let mut nodes_by_type: HashMap<String, u32> = HashMap::new();
    let mut nodes_by_source: HashMap<String, u32> = HashMap::new();

    for node in &graph.nodes {
        *nodes_by_type.entry(node.node_type.clone()).or_insert(0) += 1;

        if let Some(ref metadata) = node.metadata {
            if let Some(source) = metadata.get("source").and_then(|s| s.as_str()) {
                *nodes_by_source.entry(source.to_string()).or_insert(0) += 1;
            }
        }
    }

    Ok(GraphStats {
        total_nodes: graph.nodes.len() as u32,
        total_links: graph.links.len() as u32,
        nodes_by_type,
        nodes_by_source,
    })
}

/// Bookmarks file path in workspace
fn get_bookmarks_path() -> PathBuf {
    get_workspace_path().join("bookmarks.json")
}

/// Get list of bookmarked node IDs.
#[command]
pub async fn get_bookmarked_nodes() -> Result<Vec<String>, String> {
    let path = get_bookmarks_path();

    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read bookmarks: {}", e))?;

    let bookmarks: Vec<String> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse bookmarks: {}", e))?;

    Ok(bookmarks)
}

/// Add a node to bookmarks.
#[command]
pub async fn bookmark_node(node_id: String) -> Result<(), String> {
    log::info!("Bookmarking node: {}", node_id);

    let path = get_bookmarks_path();
    let mut bookmarks = get_bookmarked_nodes().await.unwrap_or_default();

    if !bookmarks.contains(&node_id) {
        bookmarks.push(node_id);

        let content = serde_json::to_string_pretty(&bookmarks)
            .map_err(|e| format!("Failed to serialize bookmarks: {}", e))?;

        std::fs::write(&path, content)
            .map_err(|e| format!("Failed to write bookmarks: {}", e))?;
    }

    Ok(())
}

/// Remove a node from bookmarks.
#[command]
pub async fn unbookmark_node(node_id: String) -> Result<(), String> {
    log::info!("Unbookmarking node: {}", node_id);

    let path = get_bookmarks_path();
    let mut bookmarks = get_bookmarked_nodes().await.unwrap_or_default();

    bookmarks.retain(|id| id != &node_id);

    let content = serde_json::to_string_pretty(&bookmarks)
        .map_err(|e| format!("Failed to serialize bookmarks: {}", e))?;

    std::fs::write(&path, content)
        .map_err(|e| format!("Failed to write bookmarks: {}", e))?;

    Ok(())
}
