"""
Streamlit UI for Research Agent with dynamic progress tracking and consistent styling
"""

import streamlit as st
import time
from review_agent import (
    PlanNode, TavilySearchNode, WikipediaNode, MergeResultsNode,
    DraftNode, CritiqueNode, RefineNode, EvaluationNode, OutputNode,
    build_research_agent_graph, run_research_agent
)

# Define color scheme
COLORS = {
    'primary': '#0066cc',    # Main blue
    'secondary': '#e6f3ff',  # Light blue background
    'text': '#333333',       # Dark gray for text
    'success': '#004d99',    # Darker blue for success
    'warning': '#cc0000',    # Red for warnings
}

def set_custom_style():
    """Apply custom styling to the Streamlit app"""
    st.markdown("""
        <style>
        /* Default font and colors */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: #333333;
        }
        
        /* Main title with gradient blue background */
        .main-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: white !important;
            background: linear-gradient(135deg, #0066cc, #0044aa);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Research text container */
        .research-text {
            font-family: 'Poppins', sans-serif;
            color: #1a1a1a !important;
            font-weight: 500;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #0066cc;
            margin: 10px 0;
        }
        
        /* Results container */
        .results-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e6e6e6;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: #1a1a1a !important;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Progress steps */
        .progress-step {
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            background-color: #e6f3ff;
            border-left: 3px solid #0066cc;
            color: #1a1a1a;
        }
        
        /* Active step highlight */
        .current-step {
            background-color: #0066cc;
            color: white;
            border-left: 3px solid #004d99;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #0066cc, #0044aa);
            color: white;
            font-family: 'Poppins', sans-serif;
            border: none;
            border-radius: 8px;
            padding: 10px 25px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(to right, #0066cc, #0044aa);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            border: none;
            color: #1a1a1a;
        }
        
        /* Quality metrics section */
        .metrics-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #0066cc;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = None
    if 'output' not in st.session_state:
        st.session_state.output = None
    if 'progress' not in st.session_state:
        st.session_state.progress = []

class DynamicProgressTracker:
    """Tracks and updates progress dynamically based on actual node execution"""
    
    def __init__(self):
        self.steps = {
            'planner': 'Planning research approach',
            'search': 'Searching web sources',
            'wikipedia': 'Gathering Wikipedia information',
            'merger': 'Combining research results',
            'drafter': 'Creating initial draft',
            'critic': 'Analyzing content quality',
            'refiner': 'Improving content',
            'evaluator': 'Assessing final quality',
            'outputter': 'Preparing final output'
        }
        self.current_node = None
        
    def update(self, node_name: str):
        """Update progress when a new node starts execution"""
        self.current_node = node_name
        step_description = self.steps.get(node_name, node_name)
        st.session_state.progress.append(step_description)
        self.display_progress()
        
    def display_progress(self):
        """Display current progress in sidebar"""
        with st.sidebar:
            st.markdown("### 🔄 Current Progress")
            
            # Show all completed steps
            for idx, step in enumerate(st.session_state.progress):
                with st.container():
                    st.markdown(f"✅ {step}")
                    
            # Show current step with spinner
            if self.current_node and self.current_node != 'outputter':
                with st.container():
                    st.markdown(f"⏳ {self.steps.get(self.current_node, self.current_node)}...")
            
            # Progress bar
            progress = len(st.session_state.progress) / len(self.steps)
            st.progress(progress)
            
            # Show completion percentage
            st.markdown(f"**{int(progress * 100)}% Complete**")

def create_tracked_workflow():
    """Create a workflow that reports progress to the tracker"""
    workflow = build_research_agent_graph()
    tracker = DynamicProgressTracker()
    
    # Create a status container in the sidebar
    status_container = st.sidebar.empty()
    
    def node_callback(node_name: str):
        """Callback function to update progress when nodes are executed"""
        # Update the progress tracker
        tracker.update(node_name)
        
        # Get the current step description
        step_desc = tracker.steps.get(node_name, node_name)
        
        # Update the status display
        with status_container:
            st.markdown(f"**Current Step:** {step_desc}")
            
            # Show all completed steps
            for completed_step in st.session_state.progress:
                st.markdown(f"✅ {completed_step}")
            
            # Show current step with spinner
            st.markdown(f"⏳ {step_desc}")
            
            # Update progress bar
            progress = len(st.session_state.progress) / len(tracker.steps)
            st.progress(progress)
            st.markdown(f"**Progress:** {int(progress * 100)}%")
    
    # Wrap the workflow's invoke method to include progress tracking
    original_invoke = workflow.invoke
    
    def tracked_invoke(inputs):
        st.session_state.progress = []  # Reset progress
        workflow.on_node_start = node_callback  # Set the callback
        return original_invoke(inputs)
    
    workflow.invoke = tracked_invoke
    return workflow

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    set_custom_style()
    initialize_session_state()
    
    # Main content area with styled title
    st.markdown(
        '<h1 class="main-title">🔍 AI Research Assistant</h1>',
        unsafe_allow_html=True
    )
    
    # Description with proper contrast
    st.markdown(
        '<p class="research-text">Enter your research query below, and I\'ll gather and analyze information from multiple sources.</p>',
        unsafe_allow_html=True
    )
    
    # Query input with default value
    query = st.text_input(
        "Research Query",
        value="What is the impact of artificial intelligence on modern healthcare?",
        key="query_input"
    )
    
    if st.button("Start Research", key="research_button"):
        if query:
            try:
                with st.spinner("Initializing research process..."):
                    # Create and run tracked workflow
                    workflow = create_tracked_workflow()
                    result = workflow.invoke({"user_query": query})
                    st.session_state.output = result["final_output"]
                
                # Display results
                st.success("Research Complete! 🎉")
                
                # Display the output in a clean format
                output_container = st.container()
                with output_container:
                    st.markdown("### 📊 Research Results")
                    
                    # Handle warnings if present
                    if "⚠️" in st.session_state.output:
                        st.warning("Quality Notice: Some aspects may need attention")
                    
                    # Main content with better formatting
                    st.markdown(
                        f"""<div class='results-container'>
                        <div style='color: #1a1a1a; line-height: 1.6;'>
                        {st.session_state.output}
                        </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    # Quality metrics in expander with better styling
                    if "Evaluation Details:" in st.session_state.output:
                        with st.expander("📈 View Quality Metrics"):
                            eval_details = st.session_state.output.split("Evaluation Details:")[-1]
                            st.markdown(
                                f"""<div class='metrics-container'>
                                {eval_details}
                                </div>""",
                                unsafe_allow_html=True
                            )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a research query.")

if __name__ == "__main__":
    main() 