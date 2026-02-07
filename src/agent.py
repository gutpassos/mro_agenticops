"""Agentic capabilities - ReAct pattern and tool use."""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import re

from .llm import OllamaClient
from .query import RAGPipeline


@dataclass
class Tool:
    """Agent tool definition."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for multi-step tasks.
    
    Uses LLM to reason, plan, and execute tools iteratively.
    """
    
    def __init__(
        self,
        llm: OllamaClient,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 5
    ):
        """
        Initialize ReAct agent.
        
        Args:
            llm: Language model client
            tools: Available tools for the agent
            max_iterations: Max reasoning loops
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_iterations = max_iterations
        self.history = []
        
    def run(self, task: str) -> str:
        """
        Execute task using ReAct loop.
        
        Args:
            task: User task description
            
        Returns:
            Final answer
        """
        self.history = [{"role": "user", "content": task}]
        
        for iteration in range(self.max_iterations):
            # Reason: LLM decides next action
            thought = self._reason()
            self.history.append({"role": "assistant", "content": thought})
            
            # Parse action from thought
            action = self._parse_action(thought)
            
            if action["type"] == "answer":
                return action["content"]
            
            # Act: Execute tool
            if action["type"] == "tool":
                observation = self._execute_tool(action["tool"], action["input"])
                self.history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
        
        return "Max iterations reached without final answer."
    
    def _reason(self) -> str:
        """Generate reasoning step using LLM."""
        system_prompt = self._build_system_prompt()
        response = self.llm.chat(
            messages=[{"role": "system", "content": system_prompt}] + self.history
        )
        return response
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""Você é um agente autônomo que resolve tarefas usando raciocínio e ferramentas.

Ferramentas disponíveis:
{tools_desc}

Para cada passo:
1. Pense (Thought): Analise a situação
2. Aja (Action): Use uma ferramenta OU dê a resposta final

Formato:
Thought: [seu raciocínio]
Action: [tool_name(input)] OU Answer: [resposta final]
"""
    
    def _parse_action(self, thought: str) -> Dict[str, Any]:
        """Parse action from LLM thought."""
        # Check for Answer
        answer_match = re.search(r"Answer:\s*(.+)", thought, re.DOTALL)
        if answer_match:
            return {"type": "answer", "content": answer_match.group(1).strip()}
        
        # Check for Tool use
        tool_match = re.search(r"Action:\s*(\w+)\((.+?)\)", thought)
        if tool_match:
            return {
                "type": "tool",
                "tool": tool_match.group(1),
                "input": tool_match.group(2).strip()
            }
        
        # Default: continue reasoning
        return {"type": "continue"}
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute tool and return observation."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
        
        tool = self.tools[tool_name]
        try:
            result = tool.function(tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def add_tool(self, tool: Tool):
        """Add a tool to agent's toolkit."""
        self.tools[tool.name] = tool


class MROAgent(ReActAgent):
    """Specialized agent for MRO document queries."""
    
    def __init__(self, llm: OllamaClient, rag_pipeline: RAGPipeline):
        """Initialize with RAG tool."""
        rag_tool = Tool(
            name="search_mro_docs",
            description="Busca informações nos documentos MRO usando RAG",
            function=lambda q: rag_pipeline.query(q)["answer"],
            parameters={"query": "string"}
        )
        
        super().__init__(llm, tools=[rag_tool])


def main():
    """Demo agentic execution."""
    from .llm import OllamaClient
    from .query import RAGPipeline
    
    llm = OllamaClient()
    rag = RAGPipeline()
    
    agent = MROAgent(llm, rag)
    
    task = "Explique os 3 pilares principais do MRO e dê exemplos práticos."
    result = agent.run(task)
    
    print(f"Task: {task}\n")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
