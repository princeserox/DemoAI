"""
Research Agent with Quality Control

This script implements an AI-powered research assistant that:
1. Takes a user query and performs comprehensive research
2. Gathers information from multiple sources (Tavily Search and Wikipedia)
3. Creates and refines content with built-in quality control
4. Uses a multi-step process:
   - Planning: Break down the research task
   - Research: Gather information from multiple sources
   - Drafting: Create initial content
   - Critique & Refinement: Improve content quality
   - Evaluation: Ensure quality standards are met

The agent uses a graph-based workflow where each node represents a specific task:
- PlanNode: Creates a research strategy
- TavilySearchNode: Performs web searches
- WikipediaNode: Gathers Wikipedia information
- MergeResultsNode: Combines research results
- DraftNode: Creates initial content
- CritiqueNode: Reviews and identifies improvements
- RefineNode: Implements improvements
- EvaluationNode: Assesses quality with confidence scoring
- OutputNode: Formats final output with quality metrics

Requirements:
- OpenAI API key
- Tavily API key
- Python packages: langchain, langgraph, wikipedia-api, python-dotenv
"""

import os
import wikipedia
from typing import Any, Dict, List, TypedDict
from langgraph.graph import Graph, StateGraph, START, END
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()

# Verify API key is available
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

def call_llm(prompt: str, temperature: float = 0.1) -> str:
    chat = ChatOpenAI(
        temperature=temperature,
        model_name="gpt-3.5-turbo",  # or "gpt-4", etc.
    )
    messages = [
        SystemMessage(content="You are a helpful assistant. Follow instructions carefully."),
        HumanMessage(content=prompt)
    ]
    response = chat(messages)
    return response.content

class TavilySearchNode:
    """
    A custom node that uses TavilySearchResults to perform an internet search.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("user_query", "")
        # Create tool with API key from environment
        tool = TavilySearchResults(
            tavily_api_key=os.environ["TAVILY_API_KEY"],
            max_results=3
        )
        result = tool.invoke(query)
        return {"tavily_results": result}


class WikipediaNode:
    """
    A custom node that searches Wikipedia for the user query and returns summaries.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]: 
        query = inputs.get("user_query", "")
        results = []
        try:
            wiki_titles = wikipedia.search(query, results=3)
            for title in wiki_titles:
                try:
                    page = wikipedia.page(title)
                    results.append(page.summary)
                except wikipedia.DisambiguationError:
                    pass
                except wikipedia.PageError:
                    pass
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
        return {"wiki_results": results}
    

class PlanNode:
    """
    Generates a plan (list of sub-steps) for how to address the user's query.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_query = inputs.get("user_query")
        prompt = (
            f"You are a planning assistant. The user wants to research: {user_query}\n"
            "Break down the steps you need to take into an ordered list. Be concise."
        )
        plan = call_llm(prompt)
        return {"plan": plan}
    

class DraftNode:
    """
    Uses the plan + search results to draft an initial summary/article.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        plan = inputs.get("plan", "")
        merged_results = inputs.get("merged_results", "No search results available")
        user_query = inputs.get("user_query", "")
        
        # Wait until we have both plan and merged results
        if not plan or not merged_results:
            return {"draft": "Waiting for required inputs..."}
            
        prompt = (
            f"You are a research assistant. The user wants info about '{user_query}'.\n\n"
            f"Plan:\n{plan}\n\n"
            f"Relevant info:\n{merged_results}\n\n"
            "Write a concise, coherent summary or article covering the important points using the plan above to ensure that the article or summary is coherent and relevant."
        )
        draft = call_llm(prompt)
        return {"draft": draft}


class CritiqueNode:
    """
    Critiques the draft, using the original query and plan as additional context for evaluation.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        draft = inputs["draft"]
        user_query = inputs.get("user_query", "")
        plan = inputs.get("plan", "")
        
        prompt = (
            "You are a critique assistant. You need to evaluate the following draft "
            "based on how well it addresses the original query and follows the research plan.\n\n"
            f"Original Query: {user_query}\n\n"
            f"Research Plan:\n{plan}\n\n"
            f"Draft to Review:\n{draft}\n\n"
            "Please evaluate:\n"
            "1. How well the draft answers the original query\n"
            "2. How closely it follows the research plan\n"
            "3. Any factual errors or missing information\n"
            "4. Suggestions for improvement\n\n"
            "Provide your critique in bullet points."
        )
        critique = call_llm(prompt)
        return {"critique": critique}

class RefineNode:
    """
    Refines the draft based on the critique.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        draft = inputs["draft"]
        critique = inputs["critique"]
        prompt = (
            "You are a writing assistant. Refine the draft below based on the critique.\n\n"
            f"Draft:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            "Provide an improved draft."
        )
        refined_draft = call_llm(prompt)
        return {"refined_draft": refined_draft}

class EvaluationNode:
    """
    Uses LLM to evaluate if the refined draft meets quality standards based on
    the original query, plan, critique, and refinements.
    """
    def __init__(self, max_iterations: int = 3, min_confidence: float = 0.8):
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.current_iteration = 0
        
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.current_iteration += 1
        refined_draft = inputs["refined_draft"]
        original_critique = inputs.get("critique", "")
        user_query = inputs.get("user_query", "")
        plan = inputs.get("plan", "")
        
        prompt = (
            "You are a quality evaluation assistant. Evaluate if the refined draft meets the requirements:\n\n"
            f"Original Query: {user_query}\n"
            f"Research Plan: {plan}\n"
            f"Previous Critique: {original_critique}\n"
            f"Refined Draft: {refined_draft}\n\n"
            "Evaluate the following aspects and provide a confidence score (0.0 to 1.0) for each:\n"
            "1. Query Satisfaction: How well does it answer the original query?\n"
            "2. Plan Adherence: How well does it follow the research plan?\n"
            "3. Critique Resolution: How well were the previous critique points addressed?\n"
            "4. Overall Quality: Writing quality, coherence, and completeness\n\n"
            "Format your response as:\n"
            "Query Satisfaction: [score]\n"
            "Plan Adherence: [score]\n"
            "Critique Resolution: [score]\n"
            "Overall Quality: [score]\n"
            "Average Confidence: [average of all scores]\n"
            "Remaining Issues: [bullet points of any remaining issues]\n"
            "FINAL_DECISION: [ACCEPT if average confidence >= 0.8, RETRY if < 0.8]"
        )
        
        evaluation = call_llm(prompt)
        
        # Parse the confidence score and decision
        try:
            confidence = float([line for line in evaluation.split('\n') 
                              if 'Average Confidence:' in line][0].split(':')[1].strip())
        except:
            confidence = 0.0
            
        if "FINAL_DECISION: ACCEPT" in evaluation or confidence >= self.min_confidence:
            return {
                "evaluation": "ACCEPT",
                "confidence_score": confidence,
                "evaluation_details": evaluation
            }
        elif self.current_iteration >= self.max_iterations:
            return {
                "evaluation": "ACCEPT_WITH_WARNING",
                "confidence_score": confidence,
                "evaluation_details": f"Maximum iterations ({self.max_iterations}) reached. "
                                    f"Final confidence: {confidence}. "
                                    f"Evaluation details: {evaluation}"
            }
        else:
            return {
                "evaluation": "RETRY",
                "confidence_score": confidence,
                "evaluation_details": evaluation
            }

class OutputNode:
    """
    Produces the final output with quality assessment information.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        refined_draft = inputs["refined_draft"]
        evaluation_details = inputs.get("evaluation_details", "")
        confidence_score = inputs.get("confidence_score", 0.0)
        
        if inputs.get("evaluation") == "ACCEPT_WITH_WARNING":
            final_output = (
                "⚠️ Note: This output did not reach optimal confidence levels.\n"
                f"Confidence Score: {confidence_score:.2f}\n\n"
                f"{refined_draft}\n\n"
                f"Evaluation Details:\n{evaluation_details}"
            )
        else:
            final_output = refined_draft
            
        return {"final_output": final_output}

class MergeResultsNode:
    """
    Combines search results from the Tavily ToolNode and the WikipediaNode
    into a single string for the DraftNode to consume.
    """
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tavily_data = inputs.get("tavily_results", [])
        wiki_data = inputs.get("wiki_results", [])

        combined = []

        # Handle Tavily results - extract the content from each result
        if isinstance(tavily_data, list):
            for result in tavily_data:
                if isinstance(result, dict) and 'content' in result:
                    combined.append(result['content'])
        elif isinstance(tavily_data, dict) and 'content' in tavily_data:
            combined.append(tavily_data['content'])

        # Add Wikipedia results
        if isinstance(wiki_data, list):
            combined.extend(wiki_data)
        else:
            combined.append(str(wiki_data))

        merged_str = "\n".join(filter(None, combined))
        return {"merged_results": merged_str}

# Define state schema
class State(TypedDict):
    user_query: str
    plan: str
    tavily_results: List[str]
    wiki_results: List[str]
    merged_results: str
    draft: str
    critique: str
    refined_draft: str
    evaluation: str
    final_output: str

def build_research_agent_graph() -> Graph:
    # Create nodes 
    plan_node = PlanNode()
    tavily_node = TavilySearchNode()
    wiki_node = WikipediaNode()
    merge_node = MergeResultsNode()
    draft_node = DraftNode()
    critique_node = CritiqueNode()
    refine_node = RefineNode()
    evaluation_node = EvaluationNode()
    output_node = OutputNode()
    
    workflow = StateGraph(state_schema=State)

    # Define callback wrapper
    def wrap_node_with_callback(node, node_name):
        original_call = node.__call__

        def wrapped_call(inputs):
            if hasattr(workflow, 'on_node_start'):
                workflow.on_node_start(node_name)
            return original_call(inputs)

        node.__call__ = wrapped_call
        return node

    # Add nodes with wrapped callbacks
    nodes = {
        "planner": wrap_node_with_callback(plan_node, "planner"),
        "search": wrap_node_with_callback(tavily_node, "search"),
        "wikipedia": wrap_node_with_callback(wiki_node, "wikipedia"),
        "merger": wrap_node_with_callback(merge_node, "merger"),
        "drafter": wrap_node_with_callback(draft_node, "drafter"),
        "critic": wrap_node_with_callback(critique_node, "critic"),
        "refiner": wrap_node_with_callback(refine_node, "refiner"),
        "evaluator": wrap_node_with_callback(evaluation_node, "evaluator"),
        "outputter": wrap_node_with_callback(output_node, "outputter")
    }

    # Add nodes to workflow
    for node_name, node in nodes.items():
        workflow.add_node(node_name, node)

    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge(START, "search")
    workflow.add_edge(START, "wikipedia")
    workflow.add_edge("search", "merger")
    workflow.add_edge("wikipedia", "merger")
    workflow.add_edge("merger", "drafter")
    workflow.add_edge("planner", "drafter")
    workflow.add_edge("drafter", "critic")
    workflow.add_edge("critic", "refiner")
    workflow.add_edge("refiner", "evaluator")
    workflow.add_edge("evaluator", "outputter")
    workflow.add_edge("outputter", END)

    return workflow.compile()


def run_research_agent(user_query: str) -> str:
    # Build the workflow first
    workflow = build_research_agent_graph()
    
    # Run the graph with the user query
    result = workflow.invoke({
        "user_query": user_query
    })
    
    return result["final_output"]


