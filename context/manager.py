from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
from context.layers import (
    ContextLayer,
    SystemPromptLayer,
    SystemContextLayer,
    UserContextLayer,
    ConversationHistoryLayer
)
from config.settings import settings
import tiktoken


@dataclass
class ContextAssembly:
    system_prompt: str
    full_context: str
    token_breakdown: Dict[str, int]
    truncated: bool = False


class ContextManager:
    def __init__(self):
        self._token_encoder = tiktoken.get_encoding("cl100k_base")
        self._memory_cache: Dict[str, ContextLayer] = {}
    
    async def assemble_context(
        self,
        db_schema: str = "",
        user_info: Optional[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, Any]] = None
    ) -> ContextAssembly:
        providers = [
            SystemPromptLayer(),
            SystemContextLayer(db_schema),
            UserContextLayer(user_info),
            ConversationHistoryLayer(conversation_history)
        ]
        
        layers = await asyncio.gather(*[
            provider.get_context() for provider in providers
        ])
        
        layers_sorted = sorted(layers, key=lambda x: x.priority)
        
        total_tokens = sum(layer.token_count for layer in layers_sorted)
        truncated = False
        
        if total_tokens > settings.MAX_CONTEXT_TOKENS:
            layers_sorted = self._truncate_context(layers_sorted)
            truncated = True
        
        system_prompt_layers = layers_sorted[:2]
        system_prompt = "\n\n".join(layer.content for layer in system_prompt_layers)
        
        full_context = "\n\n".join(layer.content for layer in layers_sorted)
        
        token_breakdown = {
            layer.name: layer.token_count for layer in layers_sorted
        }
        
        return ContextAssembly(
            system_prompt=system_prompt,
            full_context=full_context,
            token_breakdown=token_breakdown,
            truncated=truncated
        )
    
    def _truncate_context(self, layers: List[ContextLayer]) -> List[ContextLayer]:
        history_layer = None
        other_layers = []
        
        for layer in layers:
            if layer.name == "conversation_history":
                history_layer = layer
            else:
                other_layers.append(layer)
        
        other_tokens = sum(l.token_count for l in other_layers)
        available_tokens = settings.MAX_CONTEXT_TOKENS - other_tokens
        
        if history_layer and available_tokens > 0:
            history_text = history_layer.content
            history_tokens = self._count_tokens(history_text)
            
            if history_tokens > available_tokens:
                ratio = available_tokens / history_tokens
                lines = history_text.split("\n")
                keep_lines = max(1, int(len(lines) * ratio * 0.8))
                truncated_content = "\n".join(lines[-keep_lines:])
                
                history_layer.content = "[历史已压缩]\n" + truncated_content
                history_layer.token_count = self._count_tokens(history_layer.content)
        
        return other_layers + ([history_layer] if history_layer else [])
    
    def _count_tokens(self, text: str) -> int:
        return len(self._token_encoder.encode(text))
