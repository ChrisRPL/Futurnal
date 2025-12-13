/**
 * Graph Store Tests
 *
 * Tests for the Zustand graph state management store.
 * Covers node selection, filtering, color modes, and persistence.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import {
  useGraphStore,
  isNodeTypeVisible,
  ALL_ENTITY_TYPES,
  type ColorMode,
} from '@/stores/graphStore';
import type { EntityType } from '@/types/api';

// Mock localStorage for persistence tests
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

describe('graphStore', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorageMock.clear();

    // Reset all state including preferences
    useGraphStore.setState({
      selectedNodeId: null,
      hoveredNodeId: null,
      zoomLevel: 1.0,
      centerPosition: { x: 0, y: 0 },
      visibleNodeTypes: [],
      breathingEnabled: true,
      isExpanded: false,
      colorMode: 'colored',
      layoutMode: 'force',
      sourceFilter: [],
      confidenceRange: [0, 1],
      timeRange: { start: null, end: null },
      highlightedNodeIds: [],
      bookmarkedNodeIds: [],
    });
  });

  describe('Node Selection', () => {
    it('should set selected node', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setSelectedNode('node-123');
      });

      expect(result.current.selectedNodeId).toBe('node-123');
    });

    it('should clear selection with null', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setSelectedNode('node-123');
        result.current.setSelectedNode(null);
      });

      expect(result.current.selectedNodeId).toBeNull();
    });
  });

  describe('Hover State', () => {
    it('should set hovered node', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setHoveredNode('node-456');
      });

      expect(result.current.hoveredNodeId).toBe('node-456');
    });

    it('should clear hover with null', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setHoveredNode('node-456');
        result.current.setHoveredNode(null);
      });

      expect(result.current.hoveredNodeId).toBeNull();
    });
  });

  describe('Zoom Level', () => {
    it('should set zoom level', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setZoomLevel(2.5);
      });

      expect(result.current.zoomLevel).toBe(2.5);
    });
  });

  describe('Center Position', () => {
    it('should set center position', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setCenterPosition({ x: 100, y: 200 });
      });

      expect(result.current.centerPosition).toEqual({ x: 100, y: 200 });
    });
  });

  describe('Node Type Filtering', () => {
    it('should start with all types visible (empty array)', () => {
      const { result } = renderHook(() => useGraphStore());

      expect(result.current.visibleNodeTypes).toEqual([]);
    });

    it('should toggle type from all-visible to all-except-one', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleNodeType('Email');
      });

      // After toggle from empty, should have all except 'Email'
      expect(result.current.visibleNodeTypes).not.toContain('Email');
      expect(result.current.visibleNodeTypes.length).toBe(ALL_ENTITY_TYPES.length - 1);
    });

    it('should toggle type back to visible', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleNodeType('Email');
        result.current.toggleNodeType('Email');
      });

      // After two toggles, should be back to all visible
      expect(result.current.visibleNodeTypes).toEqual([]);
    });

    it('should hide multiple types', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleNodeType('Email');
        result.current.toggleNodeType('Mailbox');
      });

      expect(result.current.visibleNodeTypes).not.toContain('Email');
      expect(result.current.visibleNodeTypes).not.toContain('Mailbox');
      expect(result.current.visibleNodeTypes.length).toBe(ALL_ENTITY_TYPES.length - 2);
    });

    it('should show all node types', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleNodeType('Email');
        result.current.toggleNodeType('Mailbox');
        result.current.showAllNodeTypes();
      });

      expect(result.current.visibleNodeTypes).toEqual([]);
    });
  });

  describe('Color Mode', () => {
    it('should default to colored mode', () => {
      const { result } = renderHook(() => useGraphStore());

      expect(result.current.colorMode).toBe('colored');
    });

    it('should toggle to monochrome', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleColorMode();
      });

      expect(result.current.colorMode).toBe('monochrome');
    });

    it('should toggle back to colored', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleColorMode();
        result.current.toggleColorMode();
      });

      expect(result.current.colorMode).toBe('colored');
    });
  });

  describe('Breathing Animation', () => {
    it('should default to enabled', () => {
      const { result } = renderHook(() => useGraphStore());

      expect(result.current.breathingEnabled).toBe(true);
    });

    it('should toggle breathing animation', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setBreathingEnabled(false);
      });

      expect(result.current.breathingEnabled).toBe(false);
    });
  });

  describe('Expanded State', () => {
    it('should default to not expanded', () => {
      const { result } = renderHook(() => useGraphStore());

      expect(result.current.isExpanded).toBe(false);
    });

    it('should set expanded state', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setExpanded(true);
      });

      expect(result.current.isExpanded).toBe(true);
    });
  });

  describe('Reset', () => {
    it('should reset transient state but keep preferences', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setSelectedNode('node-123');
        result.current.setHoveredNode('node-456');
        result.current.setZoomLevel(2.0);
        result.current.setCenterPosition({ x: 100, y: 200 });
        result.current.toggleColorMode(); // Change to monochrome
        result.current.reset();
      });

      // Transient state should be reset
      expect(result.current.selectedNodeId).toBeNull();
      expect(result.current.hoveredNodeId).toBeNull();
      expect(result.current.zoomLevel).toBe(1.0);
      expect(result.current.centerPosition).toEqual({ x: 0, y: 0 });

      // Preferences should be preserved
      expect(result.current.colorMode).toBe('monochrome');
    });
  });

  describe('Advanced Filters', () => {
    describe('Source Filter', () => {
      it('should default to empty array (all sources visible)', () => {
        const { result } = renderHook(() => useGraphStore());
        expect(result.current.sourceFilter).toEqual([]);
      });

      it('should set source filter', () => {
        const { result } = renderHook(() => useGraphStore());

        act(() => {
          result.current.setSourceFilter(['/path/to/source1', '/path/to/source2']);
        });

        expect(result.current.sourceFilter).toEqual(['/path/to/source1', '/path/to/source2']);
      });

      it('should clear source filter', () => {
        const { result } = renderHook(() => useGraphStore());

        act(() => {
          result.current.setSourceFilter(['/path/to/source']);
          result.current.setSourceFilter([]);
        });

        expect(result.current.sourceFilter).toEqual([]);
      });
    });

    describe('Confidence Range', () => {
      it('should default to [0, 1]', () => {
        const { result } = renderHook(() => useGraphStore());
        expect(result.current.confidenceRange).toEqual([0, 1]);
      });

      it('should set confidence range', () => {
        const { result } = renderHook(() => useGraphStore());

        act(() => {
          result.current.setConfidenceRange([0.5, 0.9]);
        });

        expect(result.current.confidenceRange).toEqual([0.5, 0.9]);
      });
    });

    describe('Time Range', () => {
      it('should default to null values', () => {
        const { result } = renderHook(() => useGraphStore());
        expect(result.current.timeRange).toEqual({ start: null, end: null });
      });

      it('should set time range', () => {
        const { result } = renderHook(() => useGraphStore());

        act(() => {
          result.current.setTimeRange({ start: '2024-01-01', end: '2024-12-31' });
        });

        expect(result.current.timeRange).toEqual({ start: '2024-01-01', end: '2024-12-31' });
      });

      it('should allow partial time range', () => {
        const { result } = renderHook(() => useGraphStore());

        act(() => {
          result.current.setTimeRange({ start: '2024-01-01', end: null });
        });

        expect(result.current.timeRange).toEqual({ start: '2024-01-01', end: null });
      });
    });

    describe('Clear Filters', () => {
      it('should reset all advanced filters', () => {
        const { result } = renderHook(() => useGraphStore());

        act(() => {
          result.current.setSourceFilter(['/path/to/source']);
          result.current.setConfidenceRange([0.3, 0.8]);
          result.current.setTimeRange({ start: '2024-01-01', end: '2024-12-31' });
          result.current.clearFilters();
        });

        expect(result.current.sourceFilter).toEqual([]);
        expect(result.current.confidenceRange).toEqual([0, 1]);
        expect(result.current.timeRange).toEqual({ start: null, end: null });
      });
    });
  });

  describe('Layout Mode', () => {
    it('should default to force layout', () => {
      const { result } = renderHook(() => useGraphStore());
      expect(result.current.layoutMode).toBe('force');
    });

    it('should set layout mode to timeline', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setLayoutMode('timeline');
      });

      expect(result.current.layoutMode).toBe('timeline');
    });

    it('should switch back to force layout', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setLayoutMode('timeline');
        result.current.setLayoutMode('force');
      });

      expect(result.current.layoutMode).toBe('force');
    });
  });

  describe('Highlighted Nodes', () => {
    it('should default to empty array', () => {
      const { result } = renderHook(() => useGraphStore());
      expect(result.current.highlightedNodeIds).toEqual([]);
    });

    it('should set highlighted nodes', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setHighlightedNodes(['node-1', 'node-2', 'node-3']);
      });

      expect(result.current.highlightedNodeIds).toEqual(['node-1', 'node-2', 'node-3']);
    });

    it('should clear highlights', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setHighlightedNodes(['node-1', 'node-2']);
        result.current.clearHighlights();
      });

      expect(result.current.highlightedNodeIds).toEqual([]);
    });
  });

  describe('Bookmarks', () => {
    it('should default to empty array', () => {
      const { result } = renderHook(() => useGraphStore());
      expect(result.current.bookmarkedNodeIds).toEqual([]);
    });

    it('should set bookmarked nodes', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setBookmarkedNodes(['node-1', 'node-2']);
      });

      expect(result.current.bookmarkedNodeIds).toEqual(['node-1', 'node-2']);
    });

    it('should toggle bookmark on', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.toggleBookmark('node-1');
      });

      expect(result.current.bookmarkedNodeIds).toContain('node-1');
    });

    it('should toggle bookmark off', () => {
      const { result } = renderHook(() => useGraphStore());

      act(() => {
        result.current.setBookmarkedNodes(['node-1', 'node-2']);
        result.current.toggleBookmark('node-1');
      });

      expect(result.current.bookmarkedNodeIds).not.toContain('node-1');
      expect(result.current.bookmarkedNodeIds).toContain('node-2');
    });
  });
});

describe('isNodeTypeVisible', () => {
  it('should return true for all types when array is empty', () => {
    const visibleTypes: EntityType[] = [];

    ALL_ENTITY_TYPES.forEach((type) => {
      expect(isNodeTypeVisible(type, visibleTypes)).toBe(true);
    });
  });

  it('should return true for types in the array', () => {
    const visibleTypes: EntityType[] = ['Document', 'Person', 'Event'];

    expect(isNodeTypeVisible('Document', visibleTypes)).toBe(true);
    expect(isNodeTypeVisible('Person', visibleTypes)).toBe(true);
    expect(isNodeTypeVisible('Event', visibleTypes)).toBe(true);
  });

  it('should return false for types not in the array', () => {
    const visibleTypes: EntityType[] = ['Document', 'Person'];

    expect(isNodeTypeVisible('Email', visibleTypes)).toBe(false);
    expect(isNodeTypeVisible('Mailbox', visibleTypes)).toBe(false);
  });
});
