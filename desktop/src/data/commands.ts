/**
 * Slash Command Definitions
 *
 * Phase B: Slash Commands in Chat
 *
 * These commands are available when typing "/" in the chat input.
 * They provide quick access to common actions and navigation.
 */

import {
  Search,
  FolderPlus,
  Network,
  Settings,
  Clock,
  Save,
  FileText,
  HelpCircle,
  GitBranch,
  Lightbulb,
  Globe,
  BookOpen,
  Zap,
  type LucideIcon,
} from 'lucide-react';

export interface SlashCommand {
  /** Command name (without the leading /) */
  name: string;
  /** Display label */
  label: string;
  /** Description shown in the dropdown */
  description: string;
  /** Icon component */
  icon: LucideIcon;
  /** Command category for grouping */
  category: 'navigation' | 'action' | 'query';
  /** Whether command accepts arguments */
  hasArgs?: boolean;
  /** Argument placeholder text */
  argPlaceholder?: string;
  /** Handler type - determines how the command is processed */
  handlerType: 'navigate' | 'modal' | 'chat' | 'store';
  /** For navigate type - the route to navigate to */
  route?: string;
  /** For modal type - the modal ID to open */
  modalId?: string;
  /** For chat type - whether to send as a special message */
  chatAction?: string;
  /** Keyboard shortcut hint (display only) */
  shortcut?: string;
}

/**
 * All available slash commands
 */
export const SLASH_COMMANDS: SlashCommand[] = [
  // Navigation commands
  {
    name: 'graph',
    label: 'Graph',
    description: 'Open the knowledge graph visualization',
    icon: Network,
    category: 'navigation',
    handlerType: 'navigate',
    route: '/graph',
  },
  {
    name: 'settings',
    label: 'Settings',
    description: 'Open application settings',
    icon: Settings,
    category: 'navigation',
    handlerType: 'navigate',
    route: '/settings',
  },
  {
    name: 'activity',
    label: 'Activity',
    description: 'View activity stream',
    icon: Clock,
    category: 'navigation',
    handlerType: 'navigate',
    route: '/activity',
  },
  {
    name: 'add-source',
    label: 'Add Source',
    description: 'Add a new data source',
    icon: FolderPlus,
    category: 'navigation',
    handlerType: 'navigate',
    route: '/connectors',
  },

  // Action commands
  {
    name: 'search',
    label: 'Search',
    description: 'Search your knowledge graph',
    icon: Search,
    category: 'action',
    hasArgs: true,
    argPlaceholder: '<query>',
    handlerType: 'store',
    shortcut: '⌘K',
  },
  {
    name: 'save',
    label: 'Save Insight',
    description: 'Save the current insight to your knowledge graph',
    icon: Save,
    category: 'action',
    hasArgs: true,
    argPlaceholder: '[description]',
    handlerType: 'chat',
    chatAction: 'save_insight',
  },
  {
    name: 'paper',
    label: 'Paper Search',
    description: 'Search for academic papers on a topic',
    icon: FileText,
    category: 'action',
    hasArgs: true,
    argPlaceholder: '<topic>',
    handlerType: 'chat',
    chatAction: 'paper_search',
  },

  // Query commands
  {
    name: 'causal',
    label: 'Causal Analysis',
    description: 'Find causal chains for an event or pattern',
    icon: GitBranch,
    category: 'query',
    hasArgs: true,
    argPlaceholder: '<event>',
    handlerType: 'chat',
    chatAction: 'causal_analysis',
  },
  {
    name: 'insights',
    label: 'Insights',
    description: 'Show emergent insights from your data',
    icon: Lightbulb,
    category: 'query',
    handlerType: 'chat',
    chatAction: 'show_insights',
  },
  {
    name: 'help',
    label: 'Help',
    description: 'Show all available commands',
    icon: HelpCircle,
    category: 'action',
    handlerType: 'chat',
    chatAction: 'show_help',
  },

  // Research commands
  {
    name: 'web',
    label: 'Web Search',
    description: 'Search the web and synthesize information',
    icon: Globe,
    category: 'query',
    hasArgs: true,
    argPlaceholder: '<query>',
    handlerType: 'chat',
    chatAction: 'web_search',
  },
  {
    name: 'research',
    label: 'Deep Research',
    description: 'Conduct deep research combining PKG and web sources',
    icon: BookOpen,
    category: 'query',
    hasArgs: true,
    argPlaceholder: '<topic>',
    handlerType: 'chat',
    chatAction: 'deep_research',
  },
  {
    name: 'quick',
    label: 'Quick Search',
    description: 'Quick web search without deep analysis',
    icon: Zap,
    category: 'query',
    hasArgs: true,
    argPlaceholder: '<query>',
    handlerType: 'chat',
    chatAction: 'quick_search',
  },
];

/**
 * Get commands filtered by prefix
 */
export function filterCommands(prefix: string): SlashCommand[] {
  const normalizedPrefix = prefix.toLowerCase().replace(/^\//, '');
  if (!normalizedPrefix) return SLASH_COMMANDS;

  return SLASH_COMMANDS.filter(
    (cmd) =>
      cmd.name.toLowerCase().startsWith(normalizedPrefix) ||
      cmd.label.toLowerCase().startsWith(normalizedPrefix)
  );
}

/**
 * Get a command by exact name
 */
export function getCommandByName(name: string): SlashCommand | undefined {
  const normalizedName = name.toLowerCase().replace(/^\//, '');
  return SLASH_COMMANDS.find((cmd) => cmd.name === normalizedName);
}

/**
 * Parse a slash command from input text
 * Returns { command, args } or null if not a valid command
 */
export function parseSlashCommand(
  input: string
): { command: SlashCommand; args: string } | null {
  if (!input.startsWith('/')) return null;

  const parts = input.slice(1).split(/\s+/);
  const commandName = parts[0];
  const args = parts.slice(1).join(' ');

  const command = getCommandByName(commandName);
  if (!command) return null;

  return { command, args };
}

/**
 * Format help text for all commands
 */
export function getHelpText(): string {
  const categories = {
    navigation: 'Navigation',
    action: 'Actions',
    query: 'Queries',
  };

  let help = '**Available Commands:**\n\n';

  for (const [category, label] of Object.entries(categories)) {
    const commands = SLASH_COMMANDS.filter((cmd) => cmd.category === category);
    if (commands.length === 0) continue;

    help += `**${label}**\n`;
    for (const cmd of commands) {
      const argHint = cmd.hasArgs ? ` ${cmd.argPlaceholder}` : '';
      help += `- \`/${cmd.name}${argHint}\` - ${cmd.description}\n`;
    }
    help += '\n';
  }

  return help;
}
