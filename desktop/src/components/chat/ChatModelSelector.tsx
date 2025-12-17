/**
 * Chat Model Selector Component
 *
 * Step 08: Frontend Intelligence Integration
 * Allows users to select which model to use for chat responses.
 * Shows installation status and allows downloading models.
 *
 * Based on ModelSelector but uses chatStore for state.
 */

import { useCallback, useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { ChevronDown, Cpu, Check, Download, Loader2 } from 'lucide-react';
import { ANSWER_MODELS, DEFAULT_ANSWER_MODEL } from '@/types/api';
import { ollamaApi } from '@/lib/api';
import { useChatStore } from '@/stores/chatStore';
import { cn } from '@/lib/utils';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface PullProgress {
  model: string;
  status: string;
  percent: number;
}

export function ChatModelSelector() {
  const { selectedModel, setSelectedModel } = useChatStore();
  const [installedModels, setInstalledModels] = useState<Set<string>>(new Set());
  const [pullingModel, setPullingModel] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState<PullProgress | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch installed models on mount
  useEffect(() => {
    const fetchInstalledModels = async () => {
      setIsLoading(true);
      try {
        const response = await ollamaApi.listModels();
        const installed = new Set(response.models.map((m) => m.name));
        setInstalledModels(installed);
      } catch (error) {
        console.error('Failed to fetch installed models:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchInstalledModels();
  }, []);

  // Listen for pull progress events
  useEffect(() => {
    const unlistenProgress = listen<PullProgress>('ollama-pull-progress', (event) => {
      setPullProgress(event.payload);
    });

    const unlistenComplete = listen<{ model: string; success: boolean }>(
      'ollama-pull-complete',
      (event) => {
        if (event.payload.success) {
          setInstalledModels((prev) => new Set([...prev, event.payload.model]));
        }
        setPullingModel(null);
        setPullProgress(null);
      }
    );

    return () => {
      unlistenProgress.then((fn) => fn());
      unlistenComplete.then((fn) => fn());
    };
  }, []);

  // Check if model is installed (handle tag variations)
  const isInstalled = useCallback(
    (modelId: string) => {
      // Direct match
      if (installedModels.has(modelId)) return true;

      // Check if any installed model matches
      for (const installed of installedModels) {
        if (installed === modelId || installed.startsWith(`${modelId}:`)) {
          return true;
        }
        // Also check if modelId matches without the tag
        const installedBase = installed.split(':')[0];
        const modelBase = modelId.split(':')[0];
        if (installedBase === modelBase) {
          return true;
        }
      }
      return false;
    },
    [installedModels]
  );

  // Handle model selection or download
  const handleModelClick = useCallback(
    async (modelId: string) => {
      if (isInstalled(modelId)) {
        setSelectedModel(modelId);
      } else {
        // Start downloading
        setPullingModel(modelId);
        try {
          await ollamaApi.pullModel(modelId);
          // After successful pull, select the model
          setSelectedModel(modelId);
        } catch (error) {
          console.error('Failed to pull model:', error);
          setPullingModel(null);
          setPullProgress(null);
        }
      }
    },
    [isInstalled, setSelectedModel]
  );

  // Find current model info
  const currentModel =
    ANSWER_MODELS.find((m) => m.id === selectedModel) ??
    ANSWER_MODELS.find((m) => m.id === DEFAULT_ANSWER_MODEL) ??
    ANSWER_MODELS[1]; // Fallback to Llama 3.1

  const currentModelInstalled = isInstalled(currentModel.id);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={cn(
            'flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px]',
            'bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/70',
            'transition-colors border border-white/10',
            !currentModelInstalled && 'border-yellow-500/30'
          )}
          title={
            currentModelInstalled
              ? `Model: ${currentModel.label}`
              : `Model: ${currentModel.label} (not installed)`
          }
        >
          <Cpu className="h-3 w-3" />
          <span className="max-w-[60px] truncate">{currentModel.label}</span>
          {!currentModelInstalled && !isLoading && (
            <span className="text-yellow-500/70 text-[9px]">!</span>
          )}
          <ChevronDown className="h-2.5 w-2.5 text-white/40" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-56 bg-[#1a1a1a] border-white/10"
      >
        {ANSWER_MODELS.map((model) => {
          const installed = isInstalled(model.id);
          const isPulling = pullingModel === model.id;

          return (
            <DropdownMenuItem
              key={model.id}
              onClick={() => handleModelClick(model.id)}
              disabled={isPulling}
              className={cn(
                'flex items-center justify-between cursor-pointer',
                'focus:bg-white/10 focus:text-white',
                model.id === selectedModel && 'bg-white/5',
                !installed && !isPulling && 'opacity-70'
              )}
            >
              <div className="flex flex-col flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-white/90">{model.label}</span>
                  {installed && (
                    <Check className="h-3 w-3 text-green-500 flex-shrink-0" />
                  )}
                </div>
                <span className="text-xs text-white/40">{model.hint}</span>
                {isPulling && pullProgress && (
                  <div className="mt-1 flex items-center gap-2">
                    <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 transition-all duration-300"
                        style={{ width: `${pullProgress.percent}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-white/40">
                      {pullProgress.percent.toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2 ml-2 flex-shrink-0">
                <span className="text-xs text-white/30 font-mono">
                  {model.vram}
                </span>
                {!installed && !isPulling && (
                  <Download className="h-3.5 w-3.5 text-white/40" />
                )}
                {isPulling && (
                  <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin" />
                )}
              </div>
            </DropdownMenuItem>
          );
        })}
        {isLoading && (
          <div className="px-2 py-1.5 text-xs text-white/40 text-center">
            Checking models...
          </div>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
