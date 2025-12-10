/**
 * Mock Graph Data Generator
 *
 * Generates realistic mock graph data for UI development and testing
 * when the backend returns empty data. Follows the same pattern as
 * searchApi mock data (see api.ts lines 67-164).
 */

import type { GraphData, GraphNode, GraphLink, EntityType } from '@/types/api';

/**
 * Sample labels for each entity type to create realistic-looking nodes.
 */
const SAMPLE_LABELS: Record<EntityType, string[]> = {
  Event: [
    'Project Kickoff Meeting',
    'Code Review Session',
    'Sprint Planning',
    'Team Standup',
    'Architecture Discussion',
    'Release Deployment',
    'Bug Triage',
    'Design Review',
    'Customer Demo',
    'Retrospective',
  ],
  Document: [
    'System Architecture',
    'API Documentation',
    'User Guide',
    'Meeting Notes',
    'Research Summary',
    'Technical Spec',
    'Design Document',
    'Changelog',
    'README',
    'Requirements Doc',
  ],
  Person: [
    'Engineering Lead',
    'Product Manager',
    'Designer',
    'Developer',
    'QA Engineer',
    'DevOps Engineer',
    'Tech Writer',
    'Stakeholder',
    'Mentor',
    'Contributor',
  ],
  Code: [
    'AuthService.ts',
    'UserController.ts',
    'DatabaseConfig.rs',
    'GraphComponent.tsx',
    'ApiClient.ts',
    'TestSuite.spec.ts',
    'BuildScript.sh',
    'Dockerfile',
    'Schema.prisma',
    'Routes.tsx',
  ],
  Concept: [
    'Authentication Flow',
    'Data Pipeline',
    'Caching Strategy',
    'Error Handling',
    'State Management',
    'API Design',
    'Security Model',
    'Performance',
    'Scalability',
    'Testing Strategy',
  ],
};

/**
 * Relationship types between nodes.
 */
const RELATIONSHIPS = [
  'references',
  'created_by',
  'mentions',
  'relates_to',
  'depends_on',
  'implements',
  'documents',
  'triggers',
  'enables',
  'blocks',
];

/**
 * Entity types with weighted distribution for realistic variety.
 */
const ENTITY_TYPES: EntityType[] = ['Event', 'Document', 'Person', 'Code', 'Concept'];
const TYPE_WEIGHTS = [0.25, 0.25, 0.15, 0.2, 0.15]; // Sum to 1.0

/**
 * Pick a random entity type using weighted distribution.
 */
function pickEntityType(): EntityType {
  const rand = Math.random();
  let cumulative = 0;
  for (let i = 0; i < ENTITY_TYPES.length; i++) {
    cumulative += TYPE_WEIGHTS[i];
    if (rand < cumulative) {
      return ENTITY_TYPES[i];
    }
  }
  return ENTITY_TYPES[0];
}

/**
 * Pick a random label for an entity type.
 */
function pickLabel(type: EntityType, index: number): string {
  const labels = SAMPLE_LABELS[type];
  const baseLabel = labels[index % labels.length];
  // Add suffix if we have many nodes to avoid duplicates
  if (index >= labels.length) {
    return `${baseLabel} #${Math.floor(index / labels.length) + 1}`;
  }
  return baseLabel;
}

/**
 * Generate a random timestamp within the last N days.
 */
function randomTimestamp(daysBack: number = 90): string {
  const now = Date.now();
  const msBack = Math.random() * daysBack * 24 * 60 * 60 * 1000;
  return new Date(now - msBack).toISOString();
}

/**
 * Generate mock graph data with specified node count.
 *
 * @param nodeCount - Number of nodes to generate (default: 50)
 * @returns GraphData with nodes and links
 */
export function generateMockGraphData(nodeCount: number = 50): GraphData {
  // Track node counts per type for label generation
  const typeCounts: Record<EntityType, number> = {
    Event: 0,
    Document: 0,
    Person: 0,
    Code: 0,
    Concept: 0,
  };

  // Generate nodes with initial positions near origin (0,0)
  // This prevents d3-force from randomly scattering nodes far off-screen
  const nodes: GraphNode[] = Array.from({ length: nodeCount }, (_, i) => {
    const nodeType = pickEntityType();
    const typeIndex = typeCounts[nodeType]++;

    // Seed initial positions in a small circle around origin
    // This gives the simulation a good starting point
    const angle = (i / nodeCount) * 2 * Math.PI;
    const radius = 50 + Math.random() * 100; // 50-150px from center

    return {
      id: `node-${i}`,
      label: pickLabel(nodeType, typeIndex),
      node_type: nodeType,
      timestamp: randomTimestamp(),
      metadata: {
        source: `/path/to/source-${i}.md`,
        confidence: Math.random() * 0.4 + 0.6, // 0.6-1.0
      },
      // Initial coordinates centered at origin
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    };
  });

  // Generate links (roughly 1.5-2x node count for interesting connectivity)
  const linkCount = Math.floor(nodeCount * (1.5 + Math.random() * 0.5));
  const links: GraphLink[] = [];
  const linkSet = new Set<string>(); // Prevent duplicate links

  for (let i = 0; i < linkCount; i++) {
    const sourceIdx = Math.floor(Math.random() * nodeCount);
    const targetIdx = Math.floor(Math.random() * nodeCount);

    // Skip self-links
    if (sourceIdx === targetIdx) continue;

    // Skip duplicate links
    const linkKey = `${sourceIdx}-${targetIdx}`;
    const reverseLinkKey = `${targetIdx}-${sourceIdx}`;
    if (linkSet.has(linkKey) || linkSet.has(reverseLinkKey)) continue;

    linkSet.add(linkKey);

    links.push({
      source: nodes[sourceIdx].id,
      target: nodes[targetIdx].id,
      relationship: RELATIONSHIPS[Math.floor(Math.random() * RELATIONSHIPS.length)],
      weight: Math.random() * 0.8 + 0.2, // 0.2-1.0
    });
  }

  return { nodes, links };
}

/**
 * Generate a small graph for mini-view preview.
 */
export function generateMiniMockGraphData(): GraphData {
  return generateMockGraphData(25);
}

/**
 * Generate a large graph for testing clustering.
 */
export function generateLargeMockGraphData(): GraphData {
  return generateMockGraphData(750);
}
